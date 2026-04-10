## Summary

Briefly describe what this PR does and why.

## Related Issue

Closes #___ (or "Refs #___" if partial)

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] New brain module
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Refactor (no functional change)
- [ ] Documentation update
- [ ] Test-only change
- [ ] CI / tooling

## Changes

- item one
- item two
- item three

## Test Plan

Describe how you verified this change. Include commands run and expected output.

- [ ] `pytest tests/ -q` passes
- [ ] `pytest --cov --cov-fail-under=80` passes
- [ ] `ruff check .` passes
- [ ] `mypy --strict sybilcore/` passes
- [ ] Added / updated unit tests
- [ ] Added / updated integration tests
- [ ] Manually tested against the Dominator API (if applicable)
- [ ] Manually tested integrations (if applicable)

```
<paste relevant test output here>
```

## New Brain Checklist (only if adding a brain)

- [ ] Unit tests for clean, edge, and adversarial inputs
- [ ] Integration test inside `scan_agent`
- [ ] Entry in the brain table in `README.md`
- [ ] Entry in `docs/architecture.md`
- [ ] Registered in `sybilcore/brains/__init__.py` (if default)
- [ ] Default weight justified in the PR description
- [ ] Stress test in `tests/stress/` (if adversarial robustness is claimed)

## Breaking Changes

If this PR is a breaking change, list what breaks and how downstream users should migrate.

## Screenshots / Demos

(Optional) Screenshots, GIFs, or terminal captures for UI, dashboard, or CLI changes.

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] I have added or updated the `CHANGELOG.md` under `[Unreleased]`
- [ ] Commits are atomic and use [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] No secrets, credentials, or personal data in the diff
- [ ] Documentation updated where necessary
