# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest | :x:               |

Only the current release receives security updates.

## Reporting a Vulnerability

**Preferred method:** Use GitHub's built-in private vulnerability reporting.

1. Go to the **Security** tab of this repository
2. Click **"Report a vulnerability"**
3. Fill out the advisory form with as much detail as possible

**Fallback contact:** [security@ourochronos.org](mailto:security@ourochronos.org)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Fix target:** Within 90 days of confirmed report

## Scope

The following are considered in-scope vulnerabilities:

- Authentication or authorization bypass
- Data leakage or exposure of sensitive information
- Injection attacks (SQL, command, template, etc.)
- Cross-site scripting (XSS) or cross-site request forgery (CSRF)
- Cryptographic weaknesses
- Remote code execution
- Privilege escalation

## Out of Scope

- Social engineering attacks
- Denial of service against personal/self-hosted instances
- Issues in dependencies without a demonstrated exploit path
- Attacks requiring physical access to a user's device
- Reports from automated scanners without verified impact

## Credit

Security researchers who responsibly disclose vulnerabilities will be credited in release notes, unless they prefer to remain anonymous. Let us know your preference when reporting.

## Questions

If you're unsure whether something qualifies, report it anyway — we'd rather investigate a false positive than miss a real issue.
