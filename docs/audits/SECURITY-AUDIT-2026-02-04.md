# Valence Security Audit Report

**Date:** 2026-02-04  
**Auditor:** OpenClaw Security Subagent  
**Repository:** orobobos/valence  
**Commit:** HEAD (as of audit date)  
**Scope:** Authentication, Authorization, Input Validation, Cryptography, Secrets Management, Dependencies

---

## Executive Summary

This security audit examined the Valence personal knowledge substrate codebase, focusing on authentication flows, input validation, cryptographic implementations, and secrets management. The codebase demonstrates strong security practices overall, with proper parameterized queries preventing SQL injection and a well-implemented OAuth 2.1 flow with PKCE.

**Critical Issues:** 1  
**High Issues:** 2  
**Medium Issues:** 5  
**Low Issues:** 4  

---

## Critical Issues

### CRITICAL-001: Cross-Site Scripting (XSS) in OAuth Login Page

**File:** `src/valence/server/oauth.py` (lines 447-571)  
**Severity:** Critical  
**CVSS Score:** 8.1 (High)  

**Description:**
The OAuth login page renders user-controlled values directly into HTML without escaping:

```python
# Line 449: Error message directly interpolated
error_html = f'<div class="error">{error}</div>'

# Line 558: Rendered into HTML
{error_html}

# Line 538: Client name directly interpolated
Authorize <span>{client_name}</span>
```

The `client_name` comes from dynamic client registration (user-controlled via `/oauth/register` endpoint) and `error` comes from form validation. A malicious actor could:

1. Register a client with a malicious name containing JavaScript
2. Trick a user into visiting the authorization URL
3. Execute arbitrary JavaScript in the user's browser in the context of the Valence server

**Proof of Concept:**
```bash
# Register malicious client
curl -X POST http://localhost:8420/api/v1/oauth/register \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "<script>alert(document.cookie)</script>",
    "redirect_uris": ["https://attacker.com/callback"]
  }'
```

**Remediation:**
Use HTML escaping for all user-controlled values:

```python
import html

def _login_page(params: dict[str, Any], client_name: str, error: str | None = None) -> str:
    safe_client_name = html.escape(client_name)
    error_html = ""
    if error:
        error_html = f'<div class="error">{html.escape(error)}</div>'
    # ... use safe_client_name in template
```

**Status:** Open - requires immediate fix before production deployment

---

## High Issues

### HIGH-001: Outdated PyJWT Dependency

**File:** `pyproject.toml`  
**Severity:** High  
**CVSS Score:** 7.5 (potential, depending on CVEs)  

**Description:**
The installed PyJWT version is 2.7.0, while the latest available is 2.11.0. Multiple security fixes have been released between these versions.

**Installed:** `PyJWT==2.7.0`  
**Latest:** `PyJWT==2.11.0`

While no critical CVEs were found for 2.7.0, the recommendation is to stay current with security library updates. Versions 2.8+ include various bug fixes and security hardening.

**Remediation:**
Update `pyproject.toml` to pin a minimum version:

```toml
"PyJWT>=2.10.0",
```

Then update the installed version:
```bash
pip install --upgrade PyJWT
```

**Status:** Open

---

### HIGH-002: Subprocess Command Injection Risk in Matrix Bot

**File:** `src/valence/agents/matrix_bot.py` (lines 360-395)  
**Severity:** High  
**CVSS Score:** 7.2  

**Description:**
The Matrix bot constructs a subprocess command that includes user messages in a prompt:

```python
# Line 365: User input included in prompt
prompt = f"User {sender} says: {message}"

# Line 368-371: Command construction
cmd = [
    "claude",
    "-p", prompt,
    "--output-format", "json",
    "--permission-mode", "bypassPermissions",
]

# Line 386-391: Subprocess execution
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=600,
)
```

While `subprocess.run` with a list prevents shell injection, the message content is passed directly to the Claude CLI. If the Claude CLI has any argument parsing vulnerabilities or if the prompt is logged/processed unsafely, this could be exploited.

Additionally, the `--permission-mode bypassPermissions` flag grants extensive permissions to the invoked Claude instance.

**Remediation:**
1. Sanitize/validate message content before passing to subprocess
2. Consider whether `bypassPermissions` is necessary for all use cases
3. Add input length limits
4. Consider using a safer IPC mechanism (e.g., stdin piping instead of command-line args)

```python
import shlex

MAX_MESSAGE_LENGTH = 4000

# Truncate and sanitize message
safe_message = message[:MAX_MESSAGE_LENGTH]
# Remove any shell-like characters if passing through any shell
safe_message = safe_message.replace('\x00', '')  # Remove null bytes

prompt = f"User {html.escape(sender)} says: {safe_message}"
```

**Status:** Open

---

## Medium Issues

### MED-001: In-Memory Token Stores Without Persistence

**File:** `src/valence/server/oauth_models.py` (lines 166-243)  
**Severity:** Medium  

**Description:**
Authorization codes and refresh tokens are stored in-memory dictionaries:

```python
class AuthorizationCodeStore:
    def __init__(self):
        self._codes: dict[str, AuthorizationCode] = {}

class RefreshTokenStore:
    def __init__(self):
        self._tokens: dict[str, RefreshToken] = {}
```

This means:
1. All active sessions are lost on server restart
2. No horizontal scaling support (tokens only exist on one instance)
3. No audit trail for token usage

**Remediation:**
Consider database-backed storage for refresh tokens, or document this limitation clearly for single-instance deployments.

**Status:** Open - acceptable for single-instance deployments if documented

---

### MED-002: JWT Secret Auto-Generation in Development

**File:** `src/valence/server/config.py` (lines 203-205)  
**Severity:** Medium  

**Description:**
In non-production mode, JWT secrets are auto-generated:

```python
if not self.oauth_jwt_secret:
    object.__setattr__(self, "oauth_jwt_secret", secrets.token_hex(32))
```

This means:
1. Tokens become invalid after server restart
2. Different processes may have different secrets
3. No ability to verify tokens externally

The production check (lines 186-201) does properly enforce explicit secret configuration, which is good.

**Remediation:**
Document this behavior clearly. Consider logging a warning when auto-generating secrets.

**Status:** Open - acceptable with documentation

---

### MED-003: Missing CSRF Protection on OAuth Authorize POST

**File:** `src/valence/server/oauth.py` (lines 206-259)  
**Severity:** Medium  

**Description:**
The OAuth authorization form submission relies on OAuth's `state` parameter but doesn't include additional CSRF protection for the login form itself. While the `state` parameter protects the OAuth flow, the login form could potentially be submitted via CSRF if an attacker knows valid OAuth parameters.

**Remediation:**
Add a CSRF token to the login form:

```python
import secrets

# Generate CSRF token and include in session/cookie
csrf_token = secrets.token_urlsafe(32)
# Include as hidden field in form
# Validate on POST
```

**Status:** Open

---

### MED-004: Permissive CORS Configuration

**File:** `src/valence/server/config.py` (line 86) and `src/valence/server/app.py` (lines 859-864)  
**Severity:** Medium  

**Description:**
Default CORS configuration allows all origins:

```python
allowed_origins: list[str] = Field(
    default=["*"],
    description="Allowed CORS origins",
)
```

This is acceptable for development but should be restricted in production.

**Remediation:**
Update production deployments to specify allowed origins explicitly:

```bash
VALENCE_ALLOWED_ORIGINS='["https://your-app.com"]'
```

Document this requirement in deployment documentation.

**Status:** Open - needs documentation

---

### MED-005: Rate Limiting Not Distributed

**File:** `src/valence/server/app.py` (lines 60-79)  
**Severity:** Medium  

**Description:**
Rate limiting uses an in-memory dictionary:

```python
_rate_limits: dict[str, list[float]] = defaultdict(list)
```

This doesn't work in horizontally scaled deployments.

**Remediation:**
For multi-instance deployments, consider:
1. Redis-backed rate limiting
2. Load balancer-based rate limiting
3. Document single-instance limitation

**Status:** Open - acceptable for single-instance

---

## Low Issues

### LOW-001: Missing Security Headers

**File:** `src/valence/server/app.py`  
**Severity:** Low  

**Description:**
HTML responses don't include security headers like:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy`
- `Strict-Transport-Security` (for HTTPS)

**Remediation:**
Add middleware to set security headers on responses.

**Status:** Open

---

### LOW-002: Verbose Error Messages

**File:** `src/valence/server/app.py` (lines 215-223)  
**Severity:** Low  

**Description:**
Some error responses include internal error details:

```python
return JSONResponse(
    {
        "jsonrpc": "2.0",
        "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
        "id": None,
    },
    status_code=500,
)
```

This could leak internal information to attackers.

**Remediation:**
Log full errors server-side, return generic messages to clients:

```python
logger.exception("Internal error in MCP request")
return JSONResponse(
    {
        "jsonrpc": "2.0",
        "error": {"code": -32603, "message": "Internal server error"},
        "id": None,
    },
    status_code=500,
)
```

**Status:** Open

---

### LOW-003: OAuth Password Stored in Environment

**File:** `src/valence/server/config.py` (lines 61-64)  
**Severity:** Low  

**Description:**
OAuth password is configured via environment variable `VALENCE_OAUTH_PASSWORD`. While this is a common pattern, it means the password may appear in:
- Process listings (`ps aux`)
- Environment dumps
- Container configurations

**Remediation:**
Consider supporting file-based secrets (e.g., Docker secrets, Kubernetes secrets):

```python
oauth_password_file: Path | None = Field(
    default=None,
    description="Path to file containing OAuth password",
)
```

**Status:** Open - acceptable for most deployments

---

### LOW-004: Missing Input Validation on Some Endpoints

**File:** Various federation endpoints  
**Severity:** Low  

**Description:**
Some federation endpoints accept parameters with minimal validation:
- `limit` parameter without upper bound check
- `domain` parameter without format validation

**Remediation:**
Add validation for all user-controlled parameters:

```python
limit = min(int(request.query_params.get("limit", 50)), 100)  # Cap at 100
```

**Status:** Open

---

## Good Security Practices Observed

The audit identified several positive security practices:

### SQL Injection Prevention ✅
All database queries use parameterized queries with `psycopg2`:
```python
cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
```

### Secure Token Generation ✅
Tokens are generated using `secrets` module:
```python
code = secrets.token_urlsafe(32)
raw_token = secrets.token_urlsafe(32)
```

### Token Hashing ✅
Tokens are stored as SHA-256 hashes, not plaintext:
```python
token_hash = hashlib.sha256(token.encode()).hexdigest()
```

### Constant-Time Comparison ✅
PKCE verification uses constant-time comparison:
```python
return secrets.compare_digest(computed_challenge, code_challenge)
```

### Ed25519 Key Generation ✅
Federation identity uses proper Ed25519 from cryptography library:
```python
private_key = Ed25519PrivateKey.generate()
```

### Domain Separators ✅
VRF implementation uses domain separators to prevent cross-protocol attacks:
```python
DOMAIN_SEPARATOR_VRF_PROVE = b"valence-vrf-prove-v1"
DOMAIN_SEPARATOR_VRF_HASH = b"valence-vrf-hash-v1"
```

### Table Name Allowlist ✅
Dynamic table access uses allowlist validation:
```python
VALID_TABLES = frozenset([...])
if table_name not in VALID_TABLES:
    raise ValueError(...)
```

### File Permissions ✅
Sensitive files are created with restrictive permissions:
```python
self.token_file.chmod(0o600)
```

### Production Validation ✅
JWT secret strength is validated in production:
```python
if len(self.oauth_jwt_secret) < 32:
    raise ValueError("VALENCE_OAUTH_JWT_SECRET must be at least 32 characters.")
```

---

## Recommendations Summary

### Immediate (Before Production)
1. **Fix CRITICAL-001**: Add HTML escaping to OAuth login page
2. **Fix HIGH-001**: Update PyJWT to latest version
3. **Fix HIGH-002**: Add input sanitization to Matrix bot

### Short Term (Within 30 Days)
4. Add CSRF protection to OAuth form
5. Add security headers middleware
6. Document CORS configuration requirements
7. Review and restrict error message verbosity

### Medium Term (Within 90 Days)
8. Consider database-backed token storage for scalability
9. Add Redis-backed rate limiting for multi-instance
10. Implement file-based secret support

---

## Conclusion

The Valence codebase demonstrates good security fundamentals with proper SQL parameterization, secure random number generation, and appropriate cryptographic primitives. The critical XSS vulnerability should be addressed immediately before production deployment. The high-severity issues related to dependency updates and input sanitization should be prioritized in the short term.

The overall security posture is **acceptable for development/testing** but requires the critical and high issues to be resolved before production deployment.

---

*This audit was conducted as a point-in-time review. Regular security audits are recommended as the codebase evolves.*
