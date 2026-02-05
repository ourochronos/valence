# Valence v0.2 Security Audit

**Date:** 2026-02-04
**Auditor:** Subagent (security-audit-v02)
**Scope:** Static analysis triage + manual code review
**Input:** Bandit scan (bandit-20260204.json) + source code review

---

## Executive Summary

Valence v0.2 has a **clean security posture**. The bandit scan flagged 39 findings (1 HIGH, 18 MEDIUM, 20 LOW), but **manual triage confirms all are false positives or acceptable patterns**. The codebase demonstrates strong security practices throughout.

| Severity | Bandit Found | Real Issues | False Positives |
|----------|--------------|-------------|-----------------|
| HIGH     | 1            | 0           | 1 (MD5 for IDs) |
| MEDIUM   | 18           | 0           | 18 (SQL pattern + bind defaults) |
| LOW      | 20           | 0           | 20 (random + cleanup + constants) |
| **TOTAL**| **39**       | **0**       | **39** |

**No critical or high-severity issues require immediate action.**

---

## Detailed Findings

### HIGH Severity (Bandit)

#### H1: MD5 Hash Usage — FALSE POSITIVE ✅

**Location:** `src/valence/storage/backend.py:336`
```python
self._id = backend_id or f"local-{hashlib.md5(str(base_path).encode()).hexdigest()[:8]}"
```

**Bandit says:** "Use of weak MD5 hash for security."

**Analysis:** MD5 is used solely to generate a short, deterministic backend identifier from a file path. This is **not security-related**:
- Not used for authentication, integrity, or cryptographic purposes
- Just creates a human-readable ID suffix
- Could be replaced with any hash function (or CRC32) with identical security impact

**Recommendation:** Add `usedforsecurity=False` parameter for clarity:
```python
hashlib.md5(str(base_path).encode(), usedforsecurity=False).hexdigest()[:8]
```

---

### MEDIUM Severity (Bandit)

#### M1-M10: SQL Injection Vectors — ALL FALSE POSITIVES ✅

**Locations:** 10 files in `core/`, `federation/`, `server/`

**Pattern flagged:**
```python
cur.execute(f"""
    SELECT ... FROM beliefs
    WHERE {' AND '.join(conditions)}
    LIMIT %s
""", params)
```

**Analysis:** These are **safe parameterized queries**:
1. The `conditions` array contains only hardcoded strings like `"status = 'active'"`, `"corroboration_count >= %s"`
2. All user-supplied values go through `%s` placeholders in `params`
3. The f-string joins pre-defined column filters, not user input

**Example from corroboration.py:**
```python
conditions = [
    "corroboration_count >= %s",  # Hardcoded
    "status = 'active'",           # Hardcoded
    "superseded_by_id IS NULL",    # Hardcoded
]
params = [min_count]  # User values go here
```

**Verdict:** Bandit lacks context to understand the pattern. No injection risk.

---

#### M11-M18: Binding to 0.0.0.0 — ACCEPTABLE ✅

**Locations:** `cli/router.py`, `network/router.py`, `network/seed.py`, `server/config.py`

**Analysis:** These are CLI/config defaults allowing servers to accept connections on all interfaces. This is:
- Expected behavior for network services
- Configurable by users via CLI flags or environment variables
- Not a vulnerability in itself (firewall/deployment controls access)

---

### LOW Severity (Bandit)

#### L1-L8: Standard Random Module — FALSE POSITIVES ✅

**Locations:** `network/node.py`, `network/seed.py`, `network/config.py`, `network/messages.py`, `federation/privacy.py`

**Usage patterns:**
- `random.choices()` for weighted router selection
- `random.sample()` for peer sampling
- `random.uniform()` for timing jitter
- `random.random()` for probabilistic privacy inclusion

**Analysis:** None of these require cryptographic randomness:
- Router selection: Performance optimization, not security
- Privacy sampling: Intentional probabilistic behavior for differential privacy
- Timing jitter: DoS mitigation, not secret generation

**Security-critical code correctly uses `secrets` module:**
```python
# Found in auth.py, oauth_models.py, federation/identity.py, etc.
secrets.token_hex(32)      # Token generation
secrets.token_urlsafe(32)  # OAuth codes
secrets.compare_digest()   # PKCE verification (timing-safe)
secrets.token_bytes(32)    # Cryptographic keys
```

---

#### L9-L16: Try/Except/Pass — ACCEPTABLE ✅

**Locations:** `network/node.py`, `network/router.py`, `network/seed.py`

**Analysis:** All instances are cleanup code (closing connections, unlinking temp files) where:
- Failure is expected and harmless
- Logging would add noise without value
- The operation is best-effort by design

---

#### L17-L20: "Hardcoded Passwords" — FALSE POSITIVES ✅

**Findings:**
- `"Bearer"` — OAuth token type constant
- `"vt_"` — Token prefix for identification
- `"/api/v1/oauth/token"` — Endpoint path, not a password

**Analysis:** Bandit's heuristic incorrectly matches variable names containing "token" or "password" with string values.

---

## Issues Bandit MISSED (Manual Review)

I reviewed federation, auth, and trust code for vulnerabilities bandit cannot detect:

### ✅ Authentication: SECURE

**Token handling:**
- SHA-256 hash stored, raw token never persisted
- `secrets.token_hex(32)` for generation (256 bits entropy)
- Expiration and scope checks implemented

**OAuth:**
- PKCE with S256 method enforced
- `secrets.compare_digest()` for timing-safe comparison
- JWT with configurable secret (warns if using default)

**Federation DID signatures:**
- Ed25519 signatures verified
- Timestamp checked (5-minute window)
- Nonce for replay protection
- Body hash included in signature

### ✅ Trust Boundaries: SECURE

**Imported beliefs:**
- Trust level applied to confidence: `weighted_conf["overall"] = original_overall * trust_level`
- Origin node DID and trust stored for provenance
- Duplicate detection via federation_id and content_hash

**Cross-federation domain verification:**
- DNS TXT record verification
- DID document service endpoint verification
- Cached with TTL, failed verifications cached shorter

### ✅ No Dangerous Patterns Found

| Pattern | Status |
|---------|--------|
| `eval()` / `exec()` | ❌ Not used |
| `pickle.load()` | ❌ Not used |
| `yaml.load()` without SafeLoader | ❌ Not used |
| `subprocess` with `shell=True` | ❌ Not used |
| SQL string concatenation with user input | ❌ Not used |
| Timing-vulnerable comparisons for secrets | ❌ Not found (`secrets.compare_digest` used) |

---

## Recommendations

### Immediate (Low effort, high clarity)

1. **Silence MD5 false positive:**
   ```python
   # In storage/backend.py:336
   hashlib.md5(str(base_path).encode(), usedforsecurity=False).hexdigest()[:8]
   ```

2. **Add bandit.yaml to suppress known FPs:**
   ```yaml
   skips:
     - B104  # hardcoded_bind_all_interfaces (CLI defaults)
     - B110  # try_except_pass (cleanup code)
     - B311  # random (non-crypto usage documented)
   ```

### Future Considerations

1. **Rate limiting:** Federation endpoints could benefit from per-DID rate limits
2. **Audit logging:** Consider structured security event logging for federation operations
3. **Trust decay monitoring:** Add metrics/alerts for unusual trust score changes

---

## Conclusion

Valence v0.2 passes security audit. The bandit findings are all explainable false positives. The codebase follows security best practices:

- ✅ Cryptographic operations use appropriate primitives
- ✅ Authentication is robust with proper token/signature handling
- ✅ Trust boundaries enforced during belief import
- ✅ No injection vulnerabilities in SQL or command execution
- ✅ Timing-safe comparisons where needed

**Risk Assessment: LOW**

No blocking issues for release. The codebase demonstrates security-conscious development.
