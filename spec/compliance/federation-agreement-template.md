# Federation Operator Agreement

**Valence Network — Federation Participation Terms**

*Version 1.0 — Draft*

---

## Preamble

This Federation Operator Agreement ("Agreement") governs participation in the Valence federated belief-sharing network. By operating a Valence federation node or joining a federation as an operator, you ("Operator") agree to these terms.

**Plain language summary:** You're joining a network where data flows between independent nodes. This agreement ensures everyone plays by the same rules so the system works and users are protected.

---

## 1. Definitions

**"Belief"** — A unit of knowledge stored and shared within the Valence system, as defined in the Belief Schema specification.

**"Federation"** — A cryptographic group of agents sharing beliefs within a bounded trust context.

**"Node"** — A server or infrastructure component that participates in the Valence network and stores or processes federated data.

**"Tombstone"** — A signed record indicating that specific content should be treated as deleted across the federation.

**"Personal Data"** — Any information relating to an identified or identifiable natural person, as defined under applicable data protection law.

**"Protocol"** — The Valence technical specifications, including the Federation Layer Spec and associated documents.

---

## 2. Operator Eligibility

### 2.1 Legal Capacity

Operator represents and warrants that they:

- [ ] Are a legal entity duly organized under applicable law, OR
- [ ] Are an individual with legal capacity to enter binding agreements
- [ ] Are at least 18 years of age (if individual)
- [ ] Are not subject to sanctions or legal prohibitions preventing participation

### 2.2 Technical Capacity

Operator represents they have the technical capability to:

- Operate infrastructure meeting the Protocol requirements
- Implement security controls specified in Section 5
- Respond to incidents within the timeframes specified in Section 7
- Maintain operations for at least 90 days (or provide transition notice)

---

## 3. Data Handling Obligations

### 3.1 Processing Purposes

Operator shall process federated data only for:

- Storing and serving beliefs to authorized federation members
- Computing privacy-preserving aggregates
- Enforcing federation governance rules
- Complying with legal obligations
- Legitimate security operations (logging, intrusion detection)

Operator shall NOT:

- Use federated data for advertising or profiling
- Sell or rent access to federated data
- Process data for purposes incompatible with user consent
- Attempt to de-anonymize aggregated data

### 3.2 Data Classification Compliance

Operator agrees to:

- Respect data classification levels (L0-L4) as defined in the Compliance Framework
- Block content classified as Prohibited (L4) from federation storage
- Implement PII detection for content classified as Personal (L3)
- Honor geographic restrictions configured for the federation

### 3.3 Encryption Requirements

Operator shall:

- Store all federated beliefs in encrypted form using Protocol-specified algorithms
- Never store decrypted belief content at rest (except in memory during processing)
- Protect encryption keys using hardware security modules or equivalent
- Participate in key rotation as required by Protocol

### 3.4 Retention and Deletion

Operator agrees to:

- Honor tombstone records within 72 hours of receipt
- Participate in cryptographic key revocation for deleted content
- Maintain tombstone records for the legally required period
- Not retain decrypted copies of tombstoned content
- Provide deletion verification upon request

**Deletion is effective when:**
1. Tombstone is acknowledged AND
2. Relevant encryption keys are revoked AND
3. Node confirms no decrypted copies exist

---

## 4. User Rights Support

### 4.1 Access Requests

When a data subject exercises their right of access, Operator shall:

- Respond within 30 days (or as required by applicable law)
- Provide data in a machine-readable format
- Include processing history and sharing information
- Coordinate with other federation members if needed

### 4.2 Deletion Requests

When a data subject exercises their right to erasure, Operator shall:

- Process the request within 72 hours
- Issue tombstone records as specified in Protocol
- Propagate deletion to connected nodes
- Provide verification of deletion upon request

### 4.3 Objection and Restriction

Operator shall implement mechanisms for users to:

- Object to inclusion in aggregates
- Restrict processing of their data
- Withdraw from federation membership

---

## 5. Security Requirements

### 5.1 Minimum Security Controls

Operator shall implement:

**Infrastructure:**
- [ ] Encrypted data at rest (AES-256 or equivalent)
- [ ] Encrypted data in transit (TLS 1.3 or equivalent)
- [ ] Network segmentation for federation components
- [ ] Regular security updates (within 30 days of critical patches)

**Access Control:**
- [ ] Multi-factor authentication for administrative access
- [ ] Principle of least privilege
- [ ] Access logging and review
- [ ] Separation of duties for key management

**Monitoring:**
- [ ] Intrusion detection system
- [ ] Audit logging per Compliance Framework Section 7
- [ ] Anomaly detection for unusual data access
- [ ] Log retention per applicable requirements

**Resilience:**
- [ ] Regular backups (encrypted)
- [ ] Disaster recovery plan
- [ ] Business continuity provisions

### 5.2 Security Assessments

Operator agrees to:

- Conduct annual security assessments (self or third-party)
- Provide assessment summaries upon reasonable request
- Remediate critical vulnerabilities within 30 days
- Notify federation of unresolved critical vulnerabilities

### 5.3 Penetration Testing

For operators handling Sensitive (L2) or Personal (L3) data:

- Annual penetration testing by qualified third party
- Results shared with federation governance (summary form)

---

## 6. Incident Response

### 6.1 Notification Obligations

**Critical Incidents** (data breach, key compromise, illegal content):
- Notify federation governance within 1 hour of detection
- Notify affected users within 72 hours
- Notify regulators as required by law

**High Severity Incidents** (PII exposure, unauthorized access):
- Notify federation governance within 4 hours
- Coordinate response with affected parties

**Other Incidents:**
- Document and report in monthly compliance summary

### 6.2 Cooperation

During incidents, Operator agrees to:

- Participate in coordinated response efforts
- Share relevant logs and forensic information
- Implement containment measures as directed
- Preserve evidence for investigation

### 6.3 Post-Incident

Operator shall:

- Provide root cause analysis within 30 days
- Implement remediation measures
- Participate in lessons-learned review
- Update procedures to prevent recurrence

---

## 7. Legal Compliance

### 7.1 Applicable Law

Operator shall comply with:

- Data protection laws applicable to their jurisdiction
- Data protection laws applicable to federation members' jurisdictions
- Content laws (DMCA, NetzDG, DSA, etc.) as applicable
- Export control regulations

### 7.2 Lawful Requests

When Operator receives legal requests (subpoenas, court orders):

- Evaluate validity and scope
- Notify federation governance (unless legally prohibited)
- Narrow scope to minimum necessary
- Provide only data required by valid legal process
- Document all disclosures

### 7.3 Government Access

Operator warrants they:

- Are not subject to laws requiring secret surveillance access
- Will disclose if they become subject to such laws
- Will not provide bulk access to government entities without valid process

---

## 8. Standard Contractual Clauses (EU Data Transfers)

### 8.1 Applicability

If Operator processes Personal Data of EU residents and is located outside the European Economic Area, Operator agrees to the EU Standard Contractual Clauses (Module 3: Processor to Processor) incorporated by reference.

### 8.2 Transfer Impact Assessment

Operator shall:

- Assess whether local laws affect SCC protections
- Implement supplementary measures if necessary
- Notify federation if laws change materially

---

## 9. Liability and Indemnification

### 9.1 Operator Liability

Operator is liable for:

- Security incidents caused by failure to meet Section 5 requirements
- Deletion failures caused by non-compliance with tombstone protocol
- Unauthorized processing beyond permitted purposes
- Breach of confidentiality obligations

### 9.2 Limitation

Operator is NOT liable for:

- Content shared by users (Operator is a processor, not controller)
- Incidents caused by Protocol vulnerabilities (reported in good faith)
- Deletion failures due to unreachable nodes outside Operator's control
- Third-party actions beyond Operator's reasonable control

### 9.3 Indemnification

Operator agrees to indemnify and hold harmless other federation members from claims arising from Operator's:

- Security negligence
- Unauthorized data use
- Failure to comply with deletion requests
- Breach of this Agreement

### 9.4 Insurance

Operators handling more than 10,000 users shall maintain:

- Cyber liability insurance (minimum $1M coverage)
- Errors and omissions insurance (minimum $1M coverage)

---

## 10. Term and Termination

### 10.1 Term

This Agreement is effective upon acceptance and continues until terminated.

### 10.2 Termination for Convenience

Either party may terminate with 90 days written notice.

### 10.3 Termination for Cause

Immediate termination for:

- Material security breach
- Repeated failure to honor deletion requests
- Legal prohibition on continued operation
- Bankruptcy or insolvency

### 10.4 Wind-Down Obligations

Upon termination, Operator shall:

- Provide 90 days for data migration (unless terminated for cause)
- Continue honoring tombstones during wind-down
- Transfer or securely delete all federated data
- Provide certificate of data destruction

---

## 11. Governance

### 11.1 Amendments

This Agreement may be amended by:

- Majority vote of federation governance body
- 60 days notice to all operators
- Opportunity to withdraw before effective date

### 11.2 Dispute Resolution

Disputes shall be resolved by:

1. Good faith negotiation (30 days)
2. Mediation (30 days)
3. Binding arbitration under [ARBITRATION BODY] rules

### 11.3 Governing Law

This Agreement is governed by [JURISDICTION] law, without regard to conflicts of law principles.

---

## 12. Representations and Warranties

Operator represents and warrants that:

- [ ] Information provided in this Agreement is accurate
- [ ] Operator has authority to bind their organization
- [ ] Operator will comply with all applicable laws
- [ ] Operator will maintain required security controls
- [ ] Operator will not use data for unauthorized purposes

---

## 13. Signature

By signing below (or accepting electronically), Operator agrees to all terms of this Agreement.

**Operator Information:**

| Field | Value |
|-------|-------|
| Legal Entity Name | _________________________ |
| Jurisdiction of Incorporation | _________________________ |
| Primary Contact Name | _________________________ |
| Primary Contact Email | _________________________ |
| Legal Notice Address | _________________________ |
| Node Identifier (DID) | _________________________ |

**Signature:**

___________________________ Date: _______________

Name: ___________________________

Title: ___________________________

---

## Appendix A: Technical Compliance Checklist

Operator certifies implementation of:

### Infrastructure
- [ ] Encryption at rest (specify algorithm): _____________
- [ ] Encryption in transit (specify version): _____________
- [ ] Key management system: _____________
- [ ] Backup system (encrypted): _____________

### Protocol Compliance
- [ ] Tombstone protocol version: _____________
- [ ] Key rotation supported: Yes / No
- [ ] Aggregation privacy mechanisms: _____________
- [ ] Audit logging enabled: Yes / No

### Security
- [ ] Last security assessment date: _____________
- [ ] Last penetration test date: _____________
- [ ] Vulnerability management process: Yes / No
- [ ] Incident response plan documented: Yes / No

### Compliance
- [ ] Privacy policy URL: _____________
- [ ] DPO contact (if applicable): _____________
- [ ] Data protection registration (if required): _____________

---

## Appendix B: Data Processing Details

| Category | Details |
|----------|---------|
| Data Types Processed | Beliefs, membership records, aggregates, audit logs |
| Processing Activities | Storage, aggregation, serving, deletion |
| Data Subjects | Federation members, belief contributors |
| Retention Period | Per Compliance Framework Section 7 |
| Sub-processors | [List any sub-processors] |
| Transfer Destinations | [List countries where data may be transferred] |

---

## Appendix C: Security Contact Information

For security incidents, contact:

| Role | Contact |
|------|---------|
| Operator Security Team | _________________________ |
| Federation Security Coordinator | security@[federation-domain] |
| Valence Protocol Security | security@valence.network |

---

*This template should be customized for specific federation requirements and reviewed by legal counsel before use.*
