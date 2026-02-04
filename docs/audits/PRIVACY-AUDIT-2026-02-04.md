# Valence Privacy Audit Report

**Date:** 2026-02-04  
**Auditor:** OpenClaw Privacy Audit Subagent  
**Scope:** Differential privacy, data minimization, consent mechanisms, GDPR compliance, logging & telemetry  
**Version:** 1.0

---

## Executive Summary

This audit reviewed the privacy implementation in the Valence federated belief system. Overall, the codebase shows **strong foundational privacy design** with well-thought-out specifications and implementations for differential privacy, data classification, and deletion. However, several gaps exist between specification and implementation, particularly around **consent management**, **data portability**, and **audit logging**.

### Risk Summary

| Category | Risk Level | Issues Found |
|----------|------------|--------------|
| Differential Privacy | 🟢 LOW | Minor - implementation solid |
| Data Minimization | 🟡 MEDIUM | Federation shares more than minimum |
| Consent Mechanisms | 🔴 HIGH | Major gaps in implementation |
| GDPR Compliance | 🟡 MEDIUM | Deletion works; access/portability missing |
| Logging & Telemetry | 🟡 MEDIUM | Some PII in debug logs |

---

## 1. Differential Privacy Implementation

**Files Reviewed:**
- `src/valence/federation/privacy.py`
- `spec/components/federation-layer/DIFFERENTIAL-PRIVACY.md`

### ✅ Strengths

1. **Epsilon Bounds Are Appropriate**
   - MIN_EPSILON = 0.01 (prevents utility collapse)
   - MAX_EPSILON = 3.0 (prevents privacy collapse)
   - DEFAULT_EPSILON = 1.0 (reasonable standard privacy)
   - Validation in `PrivacyConfig.__post_init__()` enforces these bounds

2. **Privacy Budget Tracking Implemented**
   - `PrivacyBudget` class correctly tracks daily epsilon/delta consumption
   - Per-topic rate limiting (3 queries/topic/day) prevents enumeration attacks
   - Per-requester rate limiting (20 queries/hour) slows adversaries
   - Federation-wide daily limit (100 queries)

3. **Noise Addition Mechanisms Correct**
   - Laplace noise for (ε, 0)-DP: `scale = sensitivity / epsilon`
   - Gaussian noise for (ε, δ)-DP: `sigma = sensitivity * sqrt(2*ln(1.25/δ)) / ε`
   - Both implementations match the mathematical definitions

4. **k-Anonymity Enforced**
   - Minimum 5 contributors (DEFAULT_MIN_CONTRIBUTORS = 5)
   - Sensitive domains auto-elevate to 10 (SENSITIVE_MIN_CONTRIBUTORS = 10)
   - Aggregation returns `None` if threshold not met

5. **Temporal Smoothing Implemented**
   - `TemporalSmoother` class phases in/out member contributions over 24 hours
   - Prevents membership inference attacks from timing

6. **Histogram Suppression**
   - Histograms only included when contributor_count ≥ 20
   - Prevents small-sample distribution inference

### ⚠️ Minor Issues

**Issue P1.1: Noisy Count Can Be Negative**
```python
# privacy.py line ~490
noisy_count = max(0, round(add_noise(true_count, 1.0, config)))
```
While clamping to 0 is correct, a negative noisy count could leak that the true count was very small. Consider using a truncated noise mechanism.

**Recommendation:** Use censored Laplace or ensure min_contributors is checked before noise addition.

**Issue P1.2: Delta Validation Too Permissive**
```python
if self.delta >= 1e-4:
    raise ValueError(f"delta must be < 10⁻⁴, got {self.delta}")
```
The spec says delta should be "cryptographically small" (10⁻⁶ to 10⁻⁹). 10⁻⁴ is too permissive.

**Recommendation:** Default to 10⁻⁶, warn if delta > 10⁻⁵.

**Issue P1.3: Budget Reset Clears Requester Budgets**
The daily budget reset clears topic budgets but intentionally preserves requester budgets. However, the comment says "hourly reset handled separately" but there's no persistent storage—requester budgets are lost on restart.

**Recommendation:** Add persistence for requester budgets or document this as a known limitation.

---

## 2. Data Minimization

**Files Reviewed:**
- `src/valence/federation/models.py` (ShareLevel, Visibility)
- `src/valence/federation/sync.py`
- `src/valence/compliance/pii_scanner.py`

### ✅ Strengths

1. **Visibility Levels Properly Defined**
   - `PRIVATE` - never shared
   - `TRUSTED` - explicit trust required
   - `FEDERATED` - shared with DP protection
   - `PUBLIC` - discoverable

2. **Share Levels Control Data Exposure**
   - `BELIEF_ONLY` - content + confidence only
   - `WITH_PROVENANCE` - + source info
   - `FULL` - + user attribution

3. **PII Scanner Blocks Sensitive Data**
   - Detects: email, phone (US/intl), SSN, credit card, IP address
   - Classification levels properly map to federation eligibility
   - L4 (prohibited) = hard block
   - L3 (personal) = soft block with force override

### 🔴 Issues

**Issue P2.1: Federation Sync Shares More Than Necessary**
```python
# sync.py _belief_to_federated()
result = {
    "id": str(row["id"]),
    "federation_id": str(row.get("federation_id") or row["id"]),
    "origin_node_did": did,
    "content": row["content"],
    "confidence": row["confidence"],
    "domain_path": row.get("domain_path", []),
    ...
}
```
The sync always includes `valid_from` and `valid_until` if present, regardless of share_level. These temporal boundaries can be identifying.

**Recommendation:** Only include `valid_from`/`valid_until` when share_level >= WITH_PROVENANCE.

**Issue P2.2: No Redaction Before Federation**
The PII scanner is used to *block* federation of L3/L4 content, but there's no option to *redact* PII and allow federation of the sanitized content.

**Recommendation:** Add `redact_and_federate` option that uses `PIIScanner.redact_text()` before sharing.

**Issue P2.3: Domain Path Reveals Topic Interests**
Domain paths (e.g., `["health", "mental_health", "depression"]`) are always shared and can reveal sensitive information about what topics a node's users are interested in.

**Recommendation:** Consider truncating domain paths or using domain hashes for FEDERATED visibility.

**Issue P2.4: Session Context Shared in Full Mode**
```python
# models.py FederatedBelief
if self.share_level == ShareLevel.FULL:
    result["attribution"] = {
        "user_did": self.user_did,
        "created_at": self.created_at.isoformat() if self.created_at else None,
    }
```
While documented, FULL share level may inadvertently expose user identity. Consider requiring explicit per-belief consent for FULL mode.

---

## 3. Consent Mechanisms

**Files Reviewed:**
- `src/valence/compliance/deletion.py`
- `src/valence/compliance/pii_scanner.py`
- `docs/PRIVACY_GUARANTEES.md` (ConsentManager spec)
- `spec/compliance/COMPLIANCE.md` (Consent Framework)

### 🔴 Critical Gaps

**Issue P3.1: ConsentManager Not Implemented**
The `PRIVACY_GUARANTEES.md` spec defines a `ConsentManager` class with:
- `get_consent_state()`
- `update_consent()`
- Automatic visibility reduction on consent withdrawal

**This class does not exist in the codebase.** Consent is currently implicit.

**Impact:** HIGH - Users cannot:
- View their current consent state
- Withdraw consent for federation
- Set default visibility preferences
- Limit share levels

**Recommendation:** Implement `src/valence/compliance/consent.py` with:
```python
class ConsentManager:
    def get_consent_state(self, user_id: str) -> ConsentState
    def update_consent(self, user_id: str, consent: ConsentUpdate) -> None
    def check_consent_for_action(self, user_id: str, action: str) -> bool
```

**Issue P3.2: No Consent Records Storage**
`spec/compliance/COMPLIANCE.md` §2 requires consent records with:
- Consent type, scope, granted_at
- IP address hash, user agent hash
- Consent text version
- Explicit acknowledgments

**No such storage exists.** The `DeletionReason.CONSENT_WITHDRAWAL` enum exists but cannot be tracked to actual consent records.

**Recommendation:** Add `consent_records` table and `ConsentRecord` model.

**Issue P3.3: Federation Join Consent Not Captured**
The spec requires users to acknowledge federation risks at join time:
```
[ ] I understand my shared beliefs will propagate to other federation members
[ ] I understand data may exist on multiple nodes I don't control
...
```

**No such flow exists.** Federation happens silently based on visibility settings.

**Recommendation:** Add federation consent capture at first federated share.

**Issue P3.4: Per-Belief Consent for L2 Content Missing**
Spec requires consent display for L2 (SENSITIVE) content:
```
Sharing to: [Federation Name] (N members across M nodes)
[ ] I consent to sharing this belief
```

**Not implemented.** The `requires_consent` flag in `ScanResult` is computed but never acted upon.

**Recommendation:** Add consent prompt in sharing flow when `requires_consent=True`.

### ⚠️ Partial Implementation

**Issue P3.5: Consent Withdrawal Exists But Is Incomplete**
The `DeletionReason.CONSENT_WITHDRAWAL` enum exists, and `delete_user_data()` can be called with this reason. However:
- No API endpoint accepts `reason=consent_withdrawal`
- No automatic belief visibility downgrade on withdrawal
- No notification to federation of consent withdrawal

---

## 4. GDPR Compliance

**Files Reviewed:**
- `src/valence/compliance/deletion.py`
- `src/valence/server/compliance_endpoints.py`
- `spec/compliance/COMPLIANCE.md`

### ✅ Implemented (Article 17 - Right to Erasure)

1. **Tombstone Records Created**
   - `Tombstone` dataclass with required fields
   - Legal basis tracking (`legal_basis` field)
   - Signature field for verification (not yet populated)

2. **Cryptographic Erasure**
   - `perform_cryptographic_erasure()` marks keys revoked
   - Content overwritten with `'[DELETED]'`
   - Embeddings nullified

3. **Federation Propagation Started**
   - `_start_tombstone_propagation()` marks tombstone for propagation
   - `acknowledged_by` dict tracks peer acknowledgments

4. **Deletion Verification**
   - `get_deletion_verification()` returns compliance report
   - Endpoint at `/api/v1/tombstones/{id}/verification`

5. **API Endpoint**
   - `DELETE /api/v1/users/{id}/data` with reason parameter

### 🔴 Not Implemented

**Issue P4.1: Article 15 - Right of Access**
No endpoint exists for users to request all data held about them.

**Required:**
- `GET /api/v1/users/{id}/data` returning all beliefs, sessions, patterns, entities linked to user
- Audit log of processing history
- Sharing history (anonymized recipients)

**Recommendation:** Implement `export_user_data(user_id)` function and endpoint.

**Issue P4.2: Article 20 - Right to Data Portability**
The spec defines `DataPortability.export_user_data()` but it's not implemented.

**Required:**
- Export in machine-readable JSON format
- Include: beliefs, sessions, patterns, trust relationships
- Format version for import compatibility

**Recommendation:** The `export_beliefs()` function in `peer_sync.py` is close but user-specific. Adapt for Article 20 compliance.

**Issue P4.3: Article 7(3) - Withdrawal of Consent**
See Consent section - consent withdrawal is not properly implemented.

**Issue P4.4: Article 30 - Records of Processing**
No records of processing activities are maintained. Required for GDPR compliance when processing at scale.

### ⚠️ Incomplete Implementation

**Issue P4.5: Deletion Propagation Not Complete**
```python
def _start_tombstone_propagation(tombstone_id: UUID) -> None:
    """Start propagating tombstone to federation peers.
    
    In a full implementation, this would:
    1. Query active federation peers
    2. Send tombstone via VFP protocol
    3. Track acknowledgments
    
    For MVP, we just mark propagation as started.
    """
```
The actual propagation to peers is not implemented—only the marking.

**Issue P4.6: Unreachable Node Handling**
The spec requires tracking unreachable nodes and their cache expiry times. This is defined in the spec but not implemented.

---

## 5. Logging & Telemetry

**Files Reviewed:**
- All Python files for logging patterns
- `spec/compliance/COMPLIANCE.md` §7 (Audit Trail Requirements)

### ✅ Strengths

1. **Standard Logging Used Consistently**
   - All modules use `logging.getLogger(__name__)`
   - No print statements for sensitive operations

2. **User IDs Hashed in Deletion Audit**
   ```python
   def _hash_user_id(user_id: str) -> str:
       return hashlib.sha256(user_id.encode()).hexdigest()[:32]
   ```

### 🔴 Issues

**Issue P5.1: PII Logged in Some Paths**
```python
# pii_scanner.py
logger.warning(
    f"Federation HARD BLOCKED: L4 content detected "
    f"({len(result.matches)} PII matches)"
)
```
While match count is logged (safe), the actual PII values could be logged elsewhere. Grep for potential issues:

```bash
grep -rn "match.value\|pii_match.value" src/valence/
```
No direct logging of PII values found, but the `PIIMatch.value` field contains raw PII and could be logged accidentally.

**Recommendation:** Remove or redact `PIIMatch.value` after initial scan; only keep `redacted_value`.

**Issue P5.2: Request Parameters Logged**
```python
# oauth.py - form data logged at DEBUG level
logger.debug(f"Token refresh: client={client_id}")
```
DEBUG logging could capture sensitive OAuth tokens or credentials.

**Recommendation:** Ensure DEBUG logging is disabled in production; review all DEBUG log statements.

**Issue P5.3: Audit Trail Not Implemented**
`spec/compliance/COMPLIANCE.md` §7 requires:
- Append-only, tamper-evident logs
- Hash chain or similar
- Cryptographically verifiable timestamps
- 7-year retention for consent, deletion, incidents

**No such audit system exists.** Current logging is standard application logging, not compliance-grade audit trails.

**Recommendation:** Implement `AuditLogger` with:
- Append-only storage (separate from application logs)
- Hash chain for integrity
- Signed timestamps

**Issue P5.4: Log Retention Policy Not Implemented**
No log rotation or retention policy is configured. Logs may be kept indefinitely (privacy risk) or deleted too soon (compliance risk).

**Required retention per spec:**
- Consent records: 7 years
- Deletion events: 7 years  
- Membership changes: 3 years
- Access requests: 3 years

---

## 6. Additional Privacy Concerns

### Issue P6.1: No Rate Limiting on PII Scanning
The PII scanner can be called repeatedly on the same content. If scanning is logged or metered, an adversary could infer content characteristics.

### Issue P6.2: Vector Clocks Reveal Activity Patterns
```python
# sync.py
clock[peer_did] = max(clock.get(peer_did, 0), sequence)
```
Vector clock values exposed to federation peers reveal relative activity levels between nodes.

### Issue P6.3: Error Messages May Leak Information
```python
# compliance_endpoints.py
return JSONResponse({"error": str(e), "success": False}, status_code=500)
```
Unfiltered exception messages could leak internal state or data.

**Recommendation:** Sanitize error messages in production responses.

---

## Recommendations Summary

### Critical (Must Fix Before Production)

| ID | Issue | Effort |
|----|-------|--------|
| P3.1 | Implement ConsentManager | High |
| P3.2 | Add consent records storage | Medium |
| P4.1 | Implement Article 15 (right of access) | Medium |
| P4.2 | Implement Article 20 (data portability) | Medium |

### High Priority

| ID | Issue | Effort |
|----|-------|--------|
| P3.3 | Federation join consent flow | Medium |
| P3.4 | Per-belief consent for L2 content | Medium |
| P4.5 | Complete tombstone propagation | High |
| P5.3 | Implement compliance-grade audit trail | High |

### Medium Priority

| ID | Issue | Effort |
|----|-------|--------|
| P1.2 | Tighten delta validation | Low |
| P2.1 | Limit temporal data in sync | Low |
| P2.3 | Domain path privacy for federation | Medium |
| P5.4 | Implement log retention policy | Medium |

### Low Priority

| ID | Issue | Effort |
|----|-------|--------|
| P1.1 | Truncated noise for counts | Low |
| P1.3 | Persist requester budgets | Low |
| P2.2 | Add redact-and-federate option | Low |
| P5.1 | Remove raw PII from PIIMatch after scan | Low |
| P6.3 | Sanitize error messages | Low |

---

## Compliance Checklist Status

Based on `spec/compliance/COMPLIANCE.md` §8:

### Before Federation Launch
- [x] Data classification system implemented
- [x] PII detection scanner deployed
- [ ] **Consent capture UI with required acknowledgments** ❌
- [ ] **Consent record storage with retention policy** ❌
- [x] Tombstone protocol implemented
- [x] Key revocation mechanism tested
- [x] Deletion verification reporting
- [ ] Federation Agreement template finalized (template exists, not integrated)
- [ ] DPA/SCC templates ready (outside code scope)
- [ ] Geographic restriction configuration (not implemented)
- [x] Incident response plan documented (in spec)
- [ ] Incident response team designated (operational, not code)
- [ ] **Audit logging infrastructure** ❌
- [ ] **Audit log backup strategy** ❌
- [ ] Privacy policy published (documentation exists)
- [ ] Terms of service published (outside code scope)

**Compliance Score: 6/16 (37.5%)** - Not ready for production with EU users.

---

## Appendix: Files Reviewed

| Path | Status |
|------|--------|
| `src/valence/federation/privacy.py` | ✅ Reviewed |
| `src/valence/compliance/deletion.py` | ✅ Reviewed |
| `src/valence/compliance/pii_scanner.py` | ✅ Reviewed |
| `src/valence/compliance/__init__.py` | ✅ Reviewed |
| `src/valence/server/compliance_endpoints.py` | ✅ Reviewed |
| `src/valence/server/config.py` | ✅ Reviewed |
| `src/valence/federation/sync.py` | ✅ Reviewed |
| `src/valence/federation/models.py` | ✅ Reviewed |
| `src/valence/federation/trust.py` | ✅ Reviewed |
| `spec/compliance/COMPLIANCE.md` | ✅ Reviewed |
| `docs/PRIVACY_GUARANTEES.md` | ✅ Reviewed |
| `spec/components/federation-layer/DIFFERENTIAL-PRIVACY.md` | ✅ Reviewed |

---

*This audit was conducted by automated review. Critical findings should be validated by human security review before remediation.*
