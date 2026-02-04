"""Security test suite for Valence.

Tests threat model mitigations and security controls based on
the audit findings in memory/audit-security.md.

Test modules:
- test_injection.py: SQL injection, command injection attempts
- test_auth_bypass.py: Auth bypass attempts, token manipulation
- test_federation_attacks.py: Malicious peer simulation, replay attacks
- test_trust_manipulation.py: Trust score gaming, Sybil attempts
- test_data_exposure.py: PII leakage, error message exposure
"""
