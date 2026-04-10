---
name: Bug report
about: Report a reproducible problem with SybilCore
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1.
2.
3.

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include full stack traces if applicable.

```
<paste stack trace here>
```

## Minimal Reproduction

```python
# Smallest possible code snippet that reproduces the problem.
from sybilcore.brains import get_default_brains
# ...
```

## Environment

- **SybilCore version:** (e.g. `0.1.0`, or git SHA)
- **Python version:** (e.g. `3.11.9`)
- **Operating system:** (e.g. macOS 14.5, Ubuntu 22.04)
- **Install source:** (`pip`, `pip install -e .`, Docker, etc.)
- **Relevant extras:** (e.g. `[all]`, `[dashboard]`)

## Affected Brains or Components

- [ ] Coefficient aggregator
- [ ] DeceptionBrain
- [ ] ResourceHoardingBrain
- [ ] SocialGraphBrain
- [ ] IntentDriftBrain
- [ ] CompromiseBrain
- [ ] Extended brain (specify): ___
- [ ] Dominator API
- [ ] CLI
- [ ] Integrations (LangChain / CrewAI)
- [ ] Documentation

## Additional Context

Screenshots, logs, configs, adversarial inputs, anything else that helps.

## Security

If this bug has security implications, **do not** fill out this template — see [SECURITY.md](../../SECURITY.md) and report privately.
