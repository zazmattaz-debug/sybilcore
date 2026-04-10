# SybilCore Documentation

Welcome to the SybilCore documentation. SybilCore is real-time behavioral
monitoring infrastructure for autonomous AI agents, inspired by the Sibyl
System from *Psycho-Pass*. Every agent gets a single number — the **Agent
Coefficient** — that decides what it is allowed to do next.

If this is your first time here, start with the [Quickstart](quickstart.md).

---

## Index

- **[Quickstart](quickstart.md)** — Install SybilCore, run your first scan,
  and start the Dominator API in under five minutes.
- **[Architecture](architecture.md)** — How the brain ensemble, coefficient
  aggregator, and Dominator API fit together. Detailed breakdown of the
  15 brain modules shipped with SybilCore.
- **[Research](research.md)** — Index of the research artifacts in
  `research/`: tournaments, ablations, calibration curves, threat models,
  and the v4 15k-iteration adversarial study.

---

## Top-Level Files

- [`README.md`](../README.md) — Project overview and feature list
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — How to set up a dev environment
  and contribute code, brains, tests, or docs
- [`SECURITY.md`](../SECURITY.md) — Responsible disclosure policy
- [`CHANGELOG.md`](../CHANGELOG.md) — Version history following
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [`LICENSE`](../LICENSE) — MIT

---

## Getting Help

- **Questions and design discussions:** GitHub Discussions
- **Bug reports and feature requests:** GitHub Issues (templates in
  `.github/ISSUE_TEMPLATE/`)
- **Security issues:** See [SECURITY.md](../SECURITY.md) — do not open
  public issues for vulnerabilities

---

## Status

SybilCore is pre-1.0. The API surface is still changing. See
[`CHANGELOG.md`](../CHANGELOG.md) for the list of pending breaking changes
before upgrading.
