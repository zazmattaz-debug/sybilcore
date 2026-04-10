# Research

SybilCore is a research-driven project. Every brain, every weight, and
every default in the shipping ensemble is justified by an experiment that
lives somewhere in the repo. This page is the index to those experiments.

If you want the one-line summary: the current default roster was selected
by the **v4 15,000-iteration adversarial tournament** described in
[`ABLATION_RESULTS.md`](../research/ABLATION_RESULTS.md) and the
`SESSION_HANDOFF_V4_*` documents at the repo root.

---

## Research Documents

All documents below live in [`research/`](../research/).

### Methodology and Design

- [`BASELINES_PLAN.md`](../research/BASELINES_PLAN.md) — Baseline
  methodology and the datasets used (AdvBench, AgentHarm, HarmBench,
  DoNotAnswer, JailbreakBench).
- [`SDK_API_DESIGN.md`](../research/SDK_API_DESIGN.md) — SDK and API
  design notes that shaped the public surface.
- [`VISIBILITY_DESIGN.md`](../research/VISIBILITY_DESIGN.md) — Design of
  the observability surface for brains.
- [`HUMAN_EVAL_FRAMEWORK.md`](../research/HUMAN_EVAL_FRAMEWORK.md) —
  Human evaluation framework for adversarial outputs.

### Threat Models and Competitive Landscape

- [`THREAT_MODEL.md`](../research/THREAT_MODEL.md) — The threats
  SybilCore is designed to detect, and the ones it explicitly is not.
- [`BLACK_HAT_MINDSET.md`](../research/BLACK_HAT_MINDSET.md) — Red-team
  perspective on evading SybilCore and what that implies for defense.
- [`COMPETITIVE_LANDSCAPE.md`](../research/COMPETITIVE_LANDSCAPE.md) —
  Prior art, adjacent tools, and how SybilCore differs.

### Results

- [`ABLATION_RESULTS.md`](../research/ABLATION_RESULTS.md) — Leave-one-out
  ablations across the brain ensemble. This is the key document behind
  the `EmbeddingBrain` deprecation in v0.2.0.
- [`ADVERSARIAL_TRAINING_RESULTS.md`](../research/ADVERSARIAL_TRAINING_RESULTS.md) —
  Results from adversarially-trained attack populations.
- [`CORRELATION_ANALYSIS.md`](../research/CORRELATION_ANALYSIS.md) —
  Brain-to-brain correlation matrix; identifies redundancy across the
  ensemble.
- [`CALIBRATION_RESULTS.md`](../research/CALIBRATION_RESULTS.md) —
  Calibration curves and learned weights per brain.
- [`LONG_HORIZON_RESULTS.md`](../research/LONG_HORIZON_RESULTS.md) —
  Long-horizon behavioral drift detection results.

### Related Work

- [`related_work.tex`](../research/related_work.tex) — LaTeX source of
  the related-work section for the forthcoming paper.
- [`smoke_test_baselines.py`](../research/smoke_test_baselines.py) —
  Runnable smoke tests for the baseline pipelines.
- [`proposed_brain_patches/`](../research/proposed_brain_patches/) — Draft
  patches for new brains under consideration.

---

## Experiment Artifacts

Raw experiment outputs live in [`experiments/`](../experiments/) as
timestamped JSON files. Each tournament, ablation, and calibration run is
reproducible from the scripts in the repo root plus the parameters baked
into the filename.

---

## Reproducibility

To reproduce the v4 tournament results:

```bash
pip install -e ".[dev]"
pytest tests/ -q               # sanity check the install
python smoke_test_baselines.py # run baseline smoke tests
# See SESSION_HANDOFF_V4_COMPLETE.md at the repo root for the full
# command sequence used to produce the 15k-iter tournament results.
```

If you find a result that does not reproduce, please open an issue with
the exact commit hash, Python version, and full command used.
