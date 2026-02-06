# GDPR Compliance

Valence is designed with privacy as a foundational principle, not an afterthought. This document explains how Valence's architecture addresses GDPR requirements through **privacy by design**.

---

## The Core Principle: You Never Lose Custody

Traditional systems collect your data, store it on their servers, and require you to request access to what's rightfully yours. Valence inverts this model entirely.

**Your beliefs live on your node. You already have them.**

---

## Article-by-Article Compliance

### Article 15: Right of Access

> *The data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed.*

**Valence's approach:** There is no central controller. Your Valence node is *your* data store. You have complete, immediate access to all your beliefs at all times.

For federated data:

| Scenario | What Remote Nodes Have | Your Access |
|----------|------------------------|-------------|
| No federation | Nothing | N/A |
| Federated beliefs | Only what you explicitly shared | Full (you signed it) |
| Aggregated data | ε-protected contributions | Cannot be extracted (by design) |

**To access your data:** Query your local Valence instance. That's it.

### Article 16: Right to Rectification

> *The data subject shall have the right to obtain from the controller without undue delay the rectification of inaccurate personal data.*

**Valence's approach:** Update beliefs on your node. Federated copies are versioned and signed — publish a new version with higher confidence or an explicit retraction.

### Article 17: Right to Erasure ("Right to be Forgotten")

> *The data subject shall have the right to obtain from the controller the erasure of personal data.*

**Valence's approach:**
- Delete beliefs from your node instantly
- Federated copies: Issue a signed deletion request propagated via the sync protocol
- Aggregates: Your contribution is already protected by differential privacy — it cannot be individually identified or removed because it was never individually stored

### Article 20: Right to Data Portability

> *The data subject shall have the right to receive the personal data concerning him or her... in a structured, commonly used and machine-readable format.*

**Valence's approach:** Your beliefs are already in a structured, machine-readable format (JSON). Export with:

```bash
valence export --format json > my_beliefs.json
```

### Article 25: Data Protection by Design and by Default

> *The controller shall implement appropriate technical and organisational measures... which are designed to implement data-protection principles.*

**Valence's approach:** This is the entire architecture:

- **Decentralized storage** — No central point of data collection
- **Self-sovereign identity (DIDs)** — You control your cryptographic keys
- **Consent chains** — Federated data includes cryptographic provenance of consent
- **Differential privacy** — Aggregate queries protect individual contributions
- **Local-first** — Default is local storage; federation is opt-in

---

## What About Aggregated Beliefs?

When your belief contributes to a federated aggregate (e.g., corroborated facts with higher confidence), it becomes collective knowledge. This is analogous to:

- Your vote in an election — the outcome is public, your individual vote is secret
- Your response in a census — statistics are published, your household data is protected

Differential privacy guarantees (ε-budget) ensure that:
1. Your individual contribution cannot be extracted from aggregates
2. The privacy budget limits how much can be learned about any individual
3. Failed queries still consume budget (prevents probing attacks)

---

## Federation & Consent

Valence implements **consent chains** for federated data:

```
Belief → Signed by your DID → Consent scope attached → Federation request → Peer verification
```

Remote nodes can only store beliefs you explicitly chose to federate, with cryptographic proof of your consent attached.

---

## Summary

| GDPR Right | Traditional Approach | Valence Approach |
|------------|---------------------|------------------|
| Access | Request from controller | Already on your node |
| Rectification | Submit correction request | Update locally, propagate |
| Erasure | Request deletion | Delete locally, propagate retraction |
| Portability | Request export | `valence export` |
| Protection by Design | Retrofit privacy controls | Architecture is the control |

---

## Contact

For privacy-related questions about the Valence protocol, open an issue at [github.com/orobobos/valence](https://github.com/orobobos/valence) or consult the [Privacy Guarantees](./PRIVACY_GUARANTEES.md) documentation.

---

*Valence: You don't need to request access to your own mind.*
