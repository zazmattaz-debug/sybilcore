# Security Policy

SybilCore is trust infrastructure for autonomous AI agents. A vulnerability here can translate directly into real harm for every system that relies on it. We take security reports seriously and will respond quickly.

---

## Supported Versions

Security patches are backported to the current minor release and the one immediately preceding it.

| Version     | Supported |
|-------------|-----------|
| 0.2.x (dev) | Yes       |
| 0.1.x       | Yes       |
| < 0.1.0     | No        |

Pre-1.0 releases may contain breaking changes between minor versions. Fixes are delivered in the next tagged release unless the issue is actively exploited, in which case we cut an emergency patch.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security problems.**

Instead, report privately through one of the following channels:

- **Email:** github@sybilcore.dev
- **GitHub Security Advisories:** https://github.com/sybilcore/sybilcore/security/advisories/new

Please include:

1. A description of the issue and its impact
2. Steps to reproduce (minimal proof of concept preferred)
3. Affected versions and environment
4. Your suggested remediation, if any
5. Whether you wish to be credited in the advisory

We will acknowledge receipt within **48 hours** and provide a full response within **7 days** with an assessment and remediation timeline. We aim to ship a fix within **30 days** for high-severity issues.

---

## Disclosure Process

1. You report the vulnerability through a private channel above.
2. We confirm the issue and begin working on a fix.
3. We coordinate a disclosure window with you (typically 30–90 days).
4. We publish a patched release and a GitHub Security Advisory with CVE where applicable.
5. We credit you in the advisory unless you prefer to remain anonymous.

We follow **coordinated disclosure**. Please give us a reasonable window to ship a fix before going public. We will not take legal action against researchers acting in good faith.

---

## In Scope

- Core library (`sybilcore/`) — brains, coefficient aggregator, event pipeline
- Dominator API (`sybilcore/dominator/`) — REST endpoints, authentication, persistence
- Official integrations (`sybilcore/integrations/`) — LangChain, CrewAI, and others
- CLI (`sybilcore` command)
- Default configuration shipped with the package
- Dependency vulnerabilities that materially affect SybilCore

---

## Out of Scope

- Vulnerabilities in third-party services SybilCore integrates with (report to the respective vendors)
- Issues requiring physical access to a production machine
- Denial-of-service via unbounded user input in self-hosted deployments (operators are responsible for rate limiting at the edge)
- Reports generated solely by automated scanners without a working proof of concept
- Social engineering of SybilCore maintainers
- Bugs in forks or modified distributions

---

## Dual-Use Notice

SybilCore is a defensive research tool. The same techniques used to detect untrustworthy agents could in principle inform efforts to evade them. We follow responsible disclosure practices and ask that you do the same. Do not publish attack techniques that bypass SybilCore without first giving us a chance to ship a defense.

---

## Hardening Guidance for Operators

If you run SybilCore in production:

- Run the Dominator API behind an authenticated reverse proxy
- Never expose the API to the public internet without TLS and an auth layer
- Rotate secrets via environment variables — never commit them
- Pin SybilCore to a specific version and monitor the changelog
- Subscribe to GitHub Security Advisories for this repo

---

## PGP Key

A PGP key for encrypted reports is available on request via github@sybilcore.dev.

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[PLACEHOLDER — PGP key will be published here once the project has a
permanent maintainer inbox. Until then, use GitHub Security Advisories
for end-to-end encrypted reports.]
-----END PGP PUBLIC KEY BLOCK-----
```

---

## Hall of Fame

Contributors who have helped improve SybilCore's security will be listed here.

Thank you for helping keep the trust layer trustworthy.
