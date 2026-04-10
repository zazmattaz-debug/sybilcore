---
name: Feature request
about: Propose a new capability, brain, or improvement
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

## Summary

One sentence describing the feature.

## Motivation

What problem are you trying to solve? What real-world agent behavior motivates this? Link to threat models, papers, or incidents if relevant.

## Proposed Design

How would this fit into SybilCore? If it is a new brain, describe:

- Detection target (what threat pattern it watches for)
- Input signals used
- Scoring approach (heuristic, statistical, model-based)
- Expected false-positive rate
- Relationship to existing brains (overlap? orthogonal? replacement?)

If it is an API or CLI change, describe the proposed interface.

## Alternatives Considered

Other approaches you thought about and why you rejected them.

## Impact

- [ ] Adds a new brain
- [ ] Changes default brain weights
- [ ] Changes the coefficient aggregator
- [ ] Adds a new API endpoint
- [ ] Breaking change
- [ ] Backwards compatible

## Research Notes

Links to papers, datasets, or prior art that support this feature.

## Willing to Implement?

- [ ] I can send a PR
- [ ] I need help with the implementation
- [ ] I just want to propose the idea

## Additional Context

Anything else that helps us evaluate the proposal.
