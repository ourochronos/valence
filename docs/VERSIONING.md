# API Versioning Strategy

Valence uses URL-based versioning for its REST API to ensure backward compatibility and clear deprecation paths.

## Current Version

**API Version: v1** (stable)

## URL Structure

All REST endpoints are prefixed with `/api/v1/`:

```
https://your-server.com/api/v1/mcp
https://your-server.com/api/v1/health
https://your-server.com/api/v1/federation/status
https://your-server.com/api/v1/beliefs/{id}/corroboration
```

### Exceptions

The following endpoints do **not** use version prefixes as they follow RFC/standard paths:

| Path | Reason |
|------|--------|
| `/` | Root info/discovery endpoint |
| `/.well-known/oauth-authorization-server` | RFC 8414 OAuth Server Metadata |
| `/.well-known/oauth-protected-resource` | RFC 9728 Protected Resource Metadata |
| `/.well-known/vfp-node-metadata` | Valence Federation Protocol discovery |
| `/.well-known/vfp-trust-anchors` | Valence Federation Protocol trust anchors |

## MCP Tool Versioning

MCP tool schemas include an `x-version` field indicating the tool schema version:

```json
{
  "name": "belief_query",
  "description": "Search beliefs...",
  "inputSchema": {
    "type": "object",
    "x-version": "1.0",
    "properties": { ... }
  }
}
```

This allows clients to detect schema changes and handle them appropriately.

## Version Discovery

The root endpoint `/` returns version information:

```json
{
  "server": "valence",
  "version": "1.0.0",
  "apiVersion": "v1",
  "protocol": "mcp",
  "protocolVersion": "2024-11-05",
  "endpoints": {
    "mcp": "/api/v1/mcp",
    "health": "/api/v1/health",
    ...
  }
}
```

## Versioning Policy

### Semantic Versioning

- **Major version** (v1 → v2): Breaking changes to request/response formats
- **Minor version** (within v1): New endpoints, new optional fields
- **Patch version**: Bug fixes, documentation updates

### Deprecation Process

1. **Announcement**: Deprecation announced in changelog and API responses
2. **Warning Period**: 6 months minimum with deprecation warnings in responses
3. **Sunset**: Old version returns 410 Gone with migration instructions

### Backward Compatibility Guarantees

Within a major version (e.g., v1):

- Existing endpoints will not be removed
- Required fields will not be added to requests
- Response fields will not be removed
- Field types will not change

New optional fields may be added to requests and responses.

## Migration Guide

When a new API version is released:

1. Check the changelog for breaking changes
2. Update base URL from `/api/v1/` to `/api/v2/`
3. Update request/response handling for changed schemas
4. Test thoroughly before switching production

## Headers

Optional headers for version negotiation (future):

```
Accept: application/vnd.valence.v1+json
X-API-Version: 1
```

Currently these are informational; URL versioning is authoritative.
