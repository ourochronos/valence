# Location Extension for Beliefs

*Geo-aware beliefs for local knowledge.*

---

## Overview

Many valuable beliefs are location-specific:
- Infrastructure reports ("pothole at X")
- Local events ("protest happening at Y")
- Business information ("restaurant Z closed")
- Environmental observations ("air quality at W")

This extension adds optional location metadata to beliefs.

---

## Data Model

### GeoLocation Structure

```typescript
interface GeoLocation {
  // Required: coordinates
  latitude: number;          // -90 to 90
  longitude: number;         // -180 to 180
  
  // Optional: precision/context
  accuracy_meters?: number;  // GPS accuracy
  altitude_meters?: number;  // If relevant
  
  // Optional: human-readable
  address?: string;          // "123 Main St, City"
  place_name?: string;       // "Central Park"
  locality?: string;         // "Manhattan"
  region?: string;           // "New York"
  country?: string;          // "US"
  
  // Optional: area vs point
  radius_meters?: number;    // For "within X meters of point"
  bounding_box?: BoundingBox; // For regions
}

interface BoundingBox {
  north: number;  // max latitude
  south: number;  // min latitude
  east: number;   // max longitude
  west: number;   // min longitude
}
```

### Belief Extension

```typescript
interface Belief {
  // ... existing fields ...
  
  // NEW: Optional location
  location?: GeoLocation;
  
  // NEW: Location relevance
  location_scope?: 'exact' | 'approximate' | 'regional' | 'global';
}
```

### Location Scope Semantics

| Scope | Meaning | Example |
|-------|---------|---------|
| `exact` | Precise point matters | "Pothole at this exact spot" |
| `approximate` | General area | "Good coffee shops around here" |
| `regional` | City/region level | "Traffic patterns in downtown" |
| `global` | Location for context only | "Weather observation from station X" |

---

## SQL Schema Extension

```sql
-- Add to beliefs table
ALTER TABLE beliefs ADD COLUMN location geography(POINT, 4326);
ALTER TABLE beliefs ADD COLUMN location_accuracy float;
ALTER TABLE beliefs ADD COLUMN location_scope varchar(20);
ALTER TABLE beliefs ADD COLUMN location_address text;
ALTER TABLE beliefs ADD COLUMN location_metadata jsonb;

-- Spatial index for geo queries
CREATE INDEX idx_beliefs_location ON beliefs USING GIST (location);

-- Combined index for common query pattern
CREATE INDEX idx_beliefs_location_domains ON beliefs USING GIST (location) 
  WHERE is_active = true;
```

---

## Query Extensions

### Spatial Filters

```typescript
interface GeoFilter {
  // Point + radius
  near?: {
    latitude: number;
    longitude: number;
    radius_meters: number;
  };
  
  // Bounding box
  within_box?: BoundingBox;
  
  // Named place (resolved to coordinates)
  place?: string;
}

// Example query
const localBeliefs = await query({
  semantic: "infrastructure problems",
  geo: {
    near: {
      latitude: 37.7749,
      longitude: -122.4194,
      radius_meters: 1000
    }
  },
  filters: {
    min_confidence: 0.6,
    domains: ["infrastructure"]
  }
});
```

### Spatial Ranking

Location can factor into ranking:

```
geo_score = 1 / (1 + distance_km / decay_km)

final_score = (
  semantic_score × w_semantic +
  confidence_score × w_confidence +
  trust_score × w_trust +
  recency_score × w_recency +
  geo_score × w_geo  // NEW
) × diversity_multiplier
```

Default `w_geo = 0` (location doesn't affect ranking unless requested).

For local queries, `w_geo` increases based on query intent.

---

## Privacy Considerations

### Location Precision Control

Agents can choose precision when sharing:
- **Exact**: Full coordinates (for public infrastructure)
- **Fuzzy**: Rounded to ~100m (for approximate location)
- **Regional**: City/neighborhood only (for privacy)
- **Hidden**: Location used for local matching but not revealed

```typescript
interface LocationVisibility {
  precision: 'exact' | 'fuzzy' | 'regional' | 'hidden';
  reveal_to?: AgentIdentity[];  // Only these agents see full precision
}
```

### Aggregation Privacy

When federating location-aware beliefs:
- Aggregate to regional level (not individual points)
- Require k-anonymity (minimum contributors per area)
- Differential privacy on location distributions

---

## Use Cases

### Local Reporting
```
Belief: "Large pothole, approximately 2ft diameter"
Location: { lat: 37.7849, lng: -122.4094, accuracy: 5 }
Scope: exact
Domains: [infrastructure, roads, soma]
```

### Regional Knowledge
```
Belief: "Best dim sum in Chinatown is at X restaurant"
Location: { lat: 37.7941, lng: -122.4078, radius: 500 }
Scope: approximate
Domains: [food, restaurants, chinatown]
```

### Environmental Monitoring
```
Belief: "Air quality index 150 (unhealthy)"
Location: { lat: 37.7749, lng: -122.4194 }
Scope: regional
Domains: [environment, air-quality, sf]
```

---

## Implementation Notes

### Mobile Capture

When capturing from mobile:
1. Request location permission
2. Get coordinates with accuracy
3. Optionally reverse-geocode for address
4. Let user adjust precision before sharing

### Verification

Location-aware verification:
- Verifier should be near the location (or have been recently)
- Distance between reporter and verifier is evidence metadata
- Multiple distant reports of same thing → higher corroboration

### Temporal Validity

Location beliefs often have short temporal validity:
- Pothole: Valid until fixed (unknown)
- Event: Valid during event time
- Price: Valid until changed

Combine location with `valid_from`/`valid_until` for full picture.

---

*Extension designed to be optional — beliefs without location work exactly as before.*
