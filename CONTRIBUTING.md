# Contributing to SybilCore

Thank you for your interest in SybilCore. This project exists to make autonomous AI agents safer to deploy at scale, and contributions of every kind — code, brains, datasets, documentation, adversarial test cases — are welcome.

## Code of Conduct

This project adheres to the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

---

## Development Setup

```bash
git clone https://github.com/sybilcore/sybilcore.git
cd sybilcore
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the test suite

```bash
pytest
pytest --cov                       # with coverage
pytest tests/brains/test_deception.py -v   # single file
```

### Lint and type check

```bash
ruff check .
ruff format .
mypy --strict sybilcore/
```

---

## Code Style

- **Python 3.11+** with full type hints (`mypy --strict` must pass).
- **Ruff** for linting and formatting (config in `pyproject.toml`).
- **Immutability:** prefer pure functions and frozen dataclasses. Never mutate inputs.
- **File length:** 200–400 lines typical, 800 max. Split by feature.
- **Function length:** under 50 lines, nesting depth under 4.
- **Errors:** explicit; never swallow exceptions silently.
- **No hardcoded secrets, paths, or magic numbers** — use constants or config.

---

## Test Requirements

- All new features ship with tests in the same PR.
- New code must reach **80% coverage minimum** (`pytest --cov --cov-fail-under=80`).
- New brains require:
  1. Unit tests for the score function across normal, edge, and adversarial inputs.
  2. At least one integration test that runs the brain inside `scan_agent`.
  3. At least one entry in `tests/stress/` if the brain claims adversarial robustness.
- Run the full suite (`pytest`) before opening a PR.

---

## Pull Request Process

1. **Open an issue first** for any non-trivial change. This avoids wasted work.
2. Fork the repo and create a feature branch: `git checkout -b feat/my-change`.
3. Make atomic commits — one logical change per commit. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
4. Run tests and linting locally.
5. Push and open a PR against `main`. Fill out the PR template completely.
6. CI must pass (Python 3.11, 3.12, 3.13).
7. A maintainer will review. Address feedback in new commits — do not force-push during review.

---

## Adding a New Brain

Brains are the core extensibility point of SybilCore. To add one:

### 1. Create the brain file

```python
# sybilcore/brains/my_brain.py
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event


class MyBrain(BaseBrain):
    @property
    def name(self) -> str:
        return "my_brain"

    @property
    def weight(self) -> float:
        return 1.0  # relative importance in the coefficient aggregator

    def score(self, events: list[Event]) -> BrainScore:
        threat = self._analyze(events)
        return BrainScore(
            brain_name=self.name,
            value=threat,         # 0..100
            confidence=0.85,      # 0..1
            reasoning="why this score",
            indicators=["pattern_x", "pattern_y"],
        )
```

### 2. Register it

Add it to `sybilcore/brains/__init__.py` inside `get_default_brains()`.

### 3. Test it

```python
# tests/brains/test_my_brain.py
def test_my_brain_clean_input():
    brain = MyBrain()
    score = brain.score([clean_event()])
    assert score.value < 20

def test_my_brain_adversarial():
    brain = MyBrain()
    score = brain.score(malicious_events())
    assert score.value > 60
```

### 4. Document it

Add a row to the brain table in `README.md` and a section in `docs/brains.md`.

### 5. Open the PR

Use the **New Brain Proposal** issue template first if the design needs discussion.

---

## Reporting Bugs

Open a GitHub issue using the **Bug Report** template. Include:
- SybilCore version
- Python version and OS
- Minimal reproduction
- Expected vs. actual behavior
- Stack trace if applicable

For security issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

---

## Questions

- **Discussions:** GitHub Discussions for design questions and ideas.
- **Issues:** for confirmed bugs and feature requests.

Thank you for helping build the trust layer for the autonomous era.
