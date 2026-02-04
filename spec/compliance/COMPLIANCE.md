# Valence Compliance Framework

*Data protection and legal compliance for federated belief sharing.*

---

## Overview

This document establishes the compliance framework for Valence federations—systems where agents share beliefs across sovereign nodes. Once data enters federation, it propagates to multiple independent operators. This reality shapes our approach:

**Core philosophy: Encryption-first + Trusted Federation with Agreements**

We can't un-ring the bell once data spreads. So we:
1. Encrypt everything by default
2. Classify what can flow where
3. Require legal agreements from federation members
4. Design deletion to actually work (cryptographic erasure)
5. Maintain audit trails for accountability

---

## 1. Data Classification

### Classification Levels

| Level | Label | Can Federate? | Example |
|-------|-------|---------------|---------|
| **L0** | Public | ✅ Freely | Published facts, public opinions |
| **L1** | Shared | ✅ With consent | Domain expertise, professional knowledge |
| **L2** | Sensitive | ⚠️ Restricted | Medical beliefs, financial positions |
| **L3** | Personal | ❌ Never auto-federate | PII, private relationships, location |
| **L4** | Prohibited | ❌ Hard block | Illegal content, doxxing, harassment |

### What Must NOT Be Federated

The following content types MUST be blocked from federation sharing:

1. **Direct PII without explicit consent**
   - Full names linked to beliefs
   - Email addresses, phone numbers
   - Physical addresses
   - Government IDs, financial account numbers

2. **Protected health information (PHI)**
   - Unless within a HIPAA-compliant federation with BAAs

3. **Content identifying minors**
   - No beliefs that could identify individuals under 18

4. **Infringing content**
   - Copyrighted material without rights
   - Trade secrets
   - Confidential business information

5. **Legally prohibited content**
   - Varies by jurisdiction (see §6)
   - Child exploitation material (absolute prohibition)
   - Terrorist content in EU
   - Defamatory content (varies)

### Classification Enforcement

```
Belief submitted for sharing
  │
  ├── Content scanner runs (PII detection, prohibited patterns)
  │   ├── L4 detected → HARD BLOCK, log for review
  │   ├── L3 detected → SOFT BLOCK, require explicit confirmation
  │   └── L2 detected → WARNING, require consent acknowledgment
  │
  ├── User classifies explicitly (optional override for L1/L2)
  │
  └── Federation config checked
      ├── Federation allows this classification? → Proceed
      └── Federation restricts this classification? → Block
```

### PII Detection

Automated scanning for:
- Regex patterns (emails, phones, SSN formats, credit cards)
- Named entity recognition (names, organizations, locations)
- Context-aware detection (addresses in context)

**False positive handling:** Users can override L3 blocks with explicit confirmation + acknowledgment of data propagation risks.

---

## 2. Consent Framework

### Consent Types

| Consent | Scope | Revocable? | Implementation |
|---------|-------|------------|----------------|
| **Federation membership** | Joining a federation | Yes | Leave federation |
| **Sharing consent** | Per-belief sharing | Limited* | Tombstone + key revocation |
| **Processing consent** | Aggregation participation | Yes | Opt-out from aggregates |
| **Cross-federation** | Bridge sharing | Yes | Disable bridge participation |

*Limited revocation: Can't un-share from nodes that already received the data, but cryptographic erasure makes content unreadable.

### Consent Capture Requirements

#### At Federation Join

User MUST acknowledge:
```
[ ] I understand my shared beliefs will propagate to other federation members
[ ] I understand data may exist on multiple nodes I don't control
[ ] I understand deletion requests use cryptographic erasure (key destruction)
[ ] I have read and accept the Federation Agreement
[ ] I am over 18 / legal age in my jurisdiction
```

#### At Belief Sharing

For L1/L2 content, system displays:
```
Sharing to: [Federation Name] (N members across M nodes)
This belief will be:
  • Encrypted with the federation group key
  • Included in privacy-preserving aggregates
  • Propagated to federation member nodes
  
[ ] I consent to sharing this belief
[ ] I understand I can request deletion but cannot guarantee erasure from all caches
```

### Consent Records

Store immutable consent records:

```typescript
interface ConsentRecord {
  id: UUID
  subject_id: DID                    // Who consented
  consent_type: ConsentType
  scope: string                      // Federation ID or belief ID
  granted_at: timestamp
  ip_address_hash: bytes             // Hashed, not raw
  user_agent_hash: bytes
  consent_text_version: string       // Version of consent language shown
  explicit_acknowledgments: string[] // Which checkboxes were checked
  revoked_at?: timestamp
  revocation_reason?: string
}
```

**Retention:** Consent records retained for 7 years after last interaction or as required by applicable law, whichever is longer.

---

## 3. Deletion Protocol

### The Challenge

Federated data exists on multiple sovereign nodes. We can't force deletion everywhere. Solution: **Cryptographic Erasure**.

### Cryptographic Erasure

Instead of deleting data, we destroy the keys needed to read it:

```
Deletion Request
  │
  ├── Validate requester has deletion rights
  │
  ├── Create Tombstone record
  │   ├── belief_id, deletion_timestamp, requester, reason
  │   └── Signed by requester + federation admin
  │
  ├── Key Revocation
  │   ├── Mark belief encryption key as revoked
  │   ├── Remove key from all key servers
  │   └── Trigger key rotation (new epoch excludes revoked key material)
  │
  ├── Propagate Tombstone
  │   ├── Federation broadcasts tombstone to all members
  │   └── Members update local indices (stop serving this content)
  │
  └── Audit Log
      └── Record deletion request, tombstone creation, propagation receipt
```

### Tombstone Structure

```typescript
interface Tombstone {
  id: UUID
  target_type: 'belief' | 'aggregate' | 'membership' | 'federation'
  target_id: UUID
  
  created_at: timestamp
  created_by: DID
  reason: DeletionReason
  
  // Legal basis (for GDPR compliance)
  legal_basis?: 'consent_withdrawal' | 'right_to_erasure' | 'legal_order' | 'policy_violation'
  
  // Verification
  signature: bytes                   // Signed by requester
  admin_countersignature?: bytes     // For policy-based deletions
  
  // Propagation tracking
  propagation_started: timestamp
  acknowledged_by: Map<DID, timestamp>  // Which nodes acknowledged
}

enum DeletionReason {
  USER_REQUEST = 'user_request'       // GDPR Article 17
  CONSENT_WITHDRAWAL = 'consent_withdrawal'
  LEGAL_ORDER = 'legal_order'         // Court order, subpoena
  POLICY_VIOLATION = 'policy_violation'
  DATA_ACCURACY = 'data_accuracy'     // Factually incorrect
  SECURITY_INCIDENT = 'security_incident'
}
```

### Deletion Request Rights

| Requester | Can Delete |
|-----------|------------|
| Belief author | Their own beliefs |
| Federation admin | Any belief in federation (policy reasons) |
| Subject of PII | Beliefs containing their PII |
| Legal authority | As specified in valid order |

### Deletion Timelines

| Region | Requirement | Our Commitment |
|--------|-------------|----------------|
| GDPR (EU) | "Without undue delay" (typically 30 days) | 72 hours for tombstone creation, 30 days for full propagation |
| CCPA (California) | 45 days | Same as GDPR |
| Other | Best effort | 30 days |

### Deletion Verification

Users can request deletion verification report:

```typescript
interface DeletionVerificationReport {
  tombstone_id: UUID
  status: 'processing' | 'complete' | 'partial'
  
  tombstone_created: timestamp
  key_revoked: timestamp
  
  propagation_status: {
    total_nodes: number
    acknowledged: number
    pending: number
    unreachable: number
  }
  
  // For nodes we can't reach
  unreachable_nodes: {
    node_id: DID
    last_contact: timestamp
    estimated_cache_expiry: timestamp  // When encrypted content becomes unreadable
  }
}
```

---

## 4. Legal Entity Requirements

### Federation Operator Obligations

Any entity operating a Valence federation node MUST:

1. **Legal Entity Status**
   - Be a registered legal entity OR
   - Be an individual with legal capacity to contract

2. **Designated Contact**
   - Provide a valid contact for legal notices
   - Respond to inquiries within 5 business days

3. **Data Processing Agreement**
   - Sign the Federation Agreement (see `federation-agreement-template.md`)
   - For EU data: Sign Standard Contractual Clauses (SCCs) if transferring outside EEA

4. **Technical Requirements**
   - Implement tombstone protocol
   - Participate in key rotation
   - Maintain audit logs per §7
   - Support deletion verification

5. **Incident Notification**
   - Report security incidents within 72 hours
   - Participate in coordinated response

### Federation Agreement Hierarchy

```
Valence Network Agreement (base layer)
  │
  ├── Federation Operator Agreement (each node operator)
  │   └── Covers: data handling, security, incident response
  │
  ├── Federation-Specific Terms (per federation)
  │   └── Covers: governance, moderation, membership criteria
  │
  └── Member Terms of Service (end users)
      └── Covers: acceptable use, consent, liability
```

### Liability Model

| Party | Responsible For |
|-------|-----------------|
| **Federation Operator** | Infrastructure security, protocol compliance, timely deletion |
| **Federation Admin** | Moderation, membership decisions, policy enforcement |
| **Individual Member** | Content they share, accuracy of their beliefs |
| **Valence Protocol** | Protocol design (not individual implementation) |

**Safe Harbor:** Operators who comply with the deletion protocol in good faith are protected from liability for content they cannot actually delete (unreachable nodes, cached data).

---

## 5. Jurisdictional Considerations

### Data Localization

Some jurisdictions require data to stay within borders:

| Jurisdiction | Requirement | Valence Approach |
|--------------|-------------|-----------------|
| EU (GDPR) | SCCs for transfers outside EEA | Federation can be configured EU-only |
| Russia | Data localization law | Federation geographic restrictions |
| China | Cross-border data rules | Separate federation instances |
| Brazil (LGPD) | Similar to GDPR | SCCs approach |

### Federation Geographic Modes

```typescript
enum GeographicMode {
  GLOBAL = 'global'               // No restrictions
  REGIONAL = 'regional'           // Restrict to specific regions
  SINGLE_JURISDICTION = 'single'  // One jurisdiction only
}

interface GeographicConfig {
  mode: GeographicMode
  allowed_regions?: string[]       // ISO 3166 country codes
  required_sccs?: boolean          // Require SCCs for membership
  data_residency?: string          // Where primary data must reside
}
```

### Key Jurisdictional Requirements

#### European Union (GDPR)

- **Legal basis:** Consent or legitimate interest
- **Data subject rights:** Access, rectification, erasure, portability
- **Transfer restrictions:** SCCs for non-EEA transfers
- **DPO requirement:** For large-scale processing
- **Breach notification:** 72 hours to supervisory authority

**Valence compliance:**
- ✅ Consent framework covers legal basis
- ✅ Deletion protocol covers erasure right
- ✅ Export function covers portability
- ⚠️ Federation operators handling EU data need DPO consideration
- ✅ Incident response covers breach notification

#### United States

**State level (patchwork):**
- CCPA/CPRA (California): Similar to GDPR for CA residents
- VCDPA (Virginia), CPA (Colorado), etc.: Expanding

**Sectoral:**
- HIPAA: Health information (special federation type needed)
- COPPA: Children under 13 (must not collect)
- FERPA: Educational records

**Valence compliance:**
- ✅ Consent covers opt-out rights
- ✅ Classification system can enforce sectoral rules
- ⚠️ HIPAA federations require additional controls

#### Content-Specific Laws

| Law | Jurisdiction | Requirement | Valence Implementation |
|-----|--------------|-------------|----------------------|
| NetzDG | Germany | 24hr removal for illegal content | Fast-track deletion path |
| DSA | EU | Transparency, content moderation | Audit trail, moderation tools |
| Online Safety Bill | UK | Duty of care | Classification + moderation |
| DMCA | US | Notice-and-takedown | Infringement reporting flow |

---

## 6. Incident Response

### Incident Classification

| Severity | Definition | Response Time |
|----------|------------|---------------|
| **Critical** | Active data breach, key compromise, illegal content exposure | 1 hour |
| **High** | PII leak, unauthorized access, deletion failure | 4 hours |
| **Medium** | Policy violation, moderation failure, partial outage | 24 hours |
| **Low** | Minor policy issues, feature bugs | Best effort |

### Response Protocol

#### Phase 1: Detection & Triage (0-1 hour)

```
Incident Detected
  │
  ├── Automated detection (anomaly monitoring)
  │   OR
  ├── User report
  │   OR
  └── External notification (researcher, legal)
  
  │
  ├── Assign severity level
  ├── Notify incident response team
  └── Create incident ticket with timestamp
```

#### Phase 2: Containment (1-4 hours)

```
For data breach:
  ├── Identify affected data scope
  ├── Revoke compromised keys
  ├── Isolate affected nodes if necessary
  └── Preserve evidence (logs, state)

For content incident:
  ├── Issue emergency tombstone
  ├── Block content at gateway level
  └── Notify affected federations
```

#### Phase 3: Notification (4-72 hours)

| Audience | When | What |
|----------|------|------|
| Affected users | ASAP | What happened, what to do |
| Federation admins | Within 4 hours | Technical details, containment status |
| Regulators (if required) | Within 72 hours | Formal breach notification |
| Public (if warranted) | After containment | Transparent disclosure |

#### Phase 4: Remediation (24 hours - 30 days)

- Root cause analysis
- System hardening
- Policy updates if needed
- User remediation (credit monitoring, etc. if appropriate)

#### Phase 5: Post-Incident (30+ days)

- Incident report publication (sanitized)
- Lessons learned
- Process improvements
- Audit trail completion

### Private Data Leak Specifics

When private data is accidentally shared to federation:

1. **Immediate:** Issue tombstone for affected content
2. **Hour 1:** Trigger emergency key rotation
3. **Hour 4:** Contact all federation nodes with urgent deletion
4. **Day 1:** Verify propagation, document unreachable nodes
5. **Day 3:** Notify affected individuals
6. **Day 7:** Deletion verification report
7. **Day 30:** Incident closure report

---

## 7. Audit Trail Requirements

### What Must Be Logged

| Event | Required Fields | Retention |
|-------|-----------------|-----------|
| Consent granted | Subject, type, timestamp, version, acknowledgments | 7 years |
| Consent revoked | Subject, type, timestamp, reason | 7 years |
| Belief shared | Author (hashed), federation, classification, timestamp | Duration of belief + 1 year |
| Belief deleted | Tombstone ID, requester, reason, propagation status | 7 years |
| Access request | Requester, scope, timestamp, response | 3 years |
| Key rotation | Federation, epoch, reason, member count | 7 years |
| Incident | All details | 7 years |
| Membership change | Member (hashed), federation, action, timestamp | 3 years |

### Audit Log Integrity

Logs MUST be:
- **Append-only:** No modification or deletion
- **Tamper-evident:** Hash chain or similar
- **Timestamped:** Cryptographically verifiable timestamps
- **Backed up:** Geographically distributed copies

```typescript
interface AuditLogEntry {
  sequence: uint64                   // Monotonic sequence number
  timestamp: timestamp               // Verifiable timestamp
  event_type: AuditEventType
  actor_hash: bytes                  // Privacy-preserving actor ID
  details: EncryptedBlob             // Encrypted event details
  previous_hash: bytes               // Hash of previous entry
  signature: bytes                   // Signed by audit system
}
```

### Audit Access

| Requester | Access Level |
|-----------|--------------|
| Data subject | Own data only (GDPR Article 15) |
| Federation admin | Federation scope, anonymized |
| Regulator | Full access with proper authority |
| Law enforcement | Valid legal process required |

### Audit Report Types

1. **Data Subject Access Report**
   - All data held about the subject
   - Processing history
   - Sharing history (anonymized recipients)

2. **Federation Compliance Report**
   - Membership changes
   - Deletion compliance rates
   - Incident summary

3. **Regulatory Report**
   - Full detail for authorized auditors
   - Includes decryption for valid legal orders

---

## 8. Implementation Checklist

### Before Federation Launch

- [ ] Data classification system implemented
- [ ] PII detection scanner deployed
- [ ] Consent capture UI with required acknowledgments
- [ ] Consent record storage with retention policy
- [ ] Tombstone protocol implemented
- [ ] Key revocation mechanism tested
- [ ] Deletion verification reporting
- [ ] Federation Agreement template finalized
- [ ] DPA/SCC templates ready for EU operators
- [ ] Geographic restriction configuration
- [ ] Incident response plan documented
- [ ] Incident response team designated
- [ ] Audit logging infrastructure
- [ ] Audit log backup strategy
- [ ] Privacy policy published
- [ ] Terms of service published

### Ongoing Operations

- [ ] Quarterly audit log review
- [ ] Annual policy review
- [ ] Deletion request SLA monitoring
- [ ] Incident response drills (annual)
- [ ] Consent record audit (annual)
- [ ] Geographic compliance verification

---

## 9. Gaps Identified in Current Specs

The following items need additional specification work:

1. **PII Detection Specification**
   - Need detailed spec for content scanning
   - False positive handling procedures
   - Multi-language PII patterns

2. **HIPAA Federation Type**
   - Business Associate Agreement integration
   - Additional technical safeguards
   - Audit requirements beyond standard

3. **Cross-Federation Data Flows**
   - Bridge compliance when federations have different rules
   - Consent inheritance across bridges
   - Deletion propagation across bridges

4. **Minor Protection**
   - Age verification mechanisms
   - Parental consent flows (if serving under-18)
   - Content restrictions

5. **Automated Decision Making**
   - GDPR Article 22 compliance
   - If beliefs are used for automated decisions about individuals

6. **Data Portability**
   - Export format specification
   - Federation-to-federation migration
   - User data download format

7. **Lawful Interception**
   - Law enforcement request handling
   - Jurisdictional conflicts
   - Transparency reporting

---

## References

- GDPR: Regulation (EU) 2016/679
- CCPA: California Civil Code §1798.100 et seq.
- Standard Contractual Clauses: Commission Decision 2021/914
- DMCA: 17 U.S.C. § 512
- Federation Layer Spec: `../components/federation-layer/SPEC.md`
- Privacy Mechanisms: `../components/federation-layer/PRIVACY.md`

---

*"Compliance is not bureaucracy—it's the price of trust in a federated world."*
