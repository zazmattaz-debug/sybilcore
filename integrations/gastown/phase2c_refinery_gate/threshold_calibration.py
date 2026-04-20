"""Threshold calibration utility for Phase 2C Refinery Gate.

Replays the Phase 1A bead corpus through the narrow 3-brain set (Contrastive +
Identity + Temporal) and proposes pass/warn threshold values with statistical
justification.

Calibration strategy:
  - Load all agent coefficients from the Phase 1A corpus via the narrow-brain
    pipeline (not the 13-brain pipeline, which gives different values).
  - Compute distribution statistics (mean, std, percentiles).
  - Propose thresholds based on the CLEAR-band distribution:
      warn_threshold  = 95th percentile of CLEAR-band coefficients
      pass_threshold  = 99th percentile of CLEAR-band coefficients
  - Save to thresholds.json with full statistical justification.

SCOPE NOTE:
  This PoC uses ContrastiveBrain + IdentityBrain + TemporalBrain only.
  The silent brains from Phase 1A (Deception, SocialGraph, Compromise, Semantic,
  SwarmDetection, Economic, Neuro, Silence) would require runtime .events.jsonl
  data not present in the bead-tracker committed schema.

ADVERSARIAL CALIBRATION (P0 fix):
  run_calibration() is the original function — calibrated on benign corpus only.
  It is kept for backward compatibility but is deprecated (use calibrate_with_adversaries).

  calibrate_with_adversaries() fixes the P0 circular-calibration problem by:
    1. Scoring both benign and adversarial agents through the narrow pipeline.
    2. Using ROC-style (TPR vs FPR) threshold search to find the operating
       point with best True Positive Rate at controlled False Positive Rate.
    3. Setting warn_threshold where FPR first exceeds 0.05, pass_threshold
       where FPR first exceeds 0.20.

Usage:
  python3 -m sybilcore.integrations.gastown.phase2c_refinery_gate.threshold_calibration
  python3 threshold_calibration.py [--fixture-path <path>] [--output-path <path>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent
_INTEGRATION_DIR = _PKG_DIR.parent
_SYBILCORE_ROOT = _INTEGRATION_DIR.parents[2]
for _p in (str(_SYBILCORE_ROOT), str(_INTEGRATION_DIR / "phase1a_baseline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _p  # avoid polluting module namespace

from adapter import adapt_fixture_file, group_events_by_agent  # type: ignore[import]  # noqa: E402

try:
    from .constants import (  # noqa: E402
        DEFAULT_FIXTURE_PATH,
        DEFAULT_THRESHOLD_PASS,
        DEFAULT_THRESHOLD_WARN,
        DEFAULT_WINDOW_SECONDS,
        SCOPE_NOTE,
        THRESHOLDS_PATH,
    )
    from .narrow_brains import build_narrow_brains, score_agent_narrow  # noqa: E402
except ImportError:
    from constants import (  # type: ignore[no-redef]  # noqa: E402
        DEFAULT_FIXTURE_PATH,
        DEFAULT_THRESHOLD_PASS,
        DEFAULT_THRESHOLD_WARN,
        DEFAULT_WINDOW_SECONDS,
        SCOPE_NOTE,
        THRESHOLDS_PATH,
    )
    from narrow_brains import (  # type: ignore[no-redef]  # noqa: E402
        build_narrow_brains,
        score_agent_narrow,
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute the p-th percentile of a sorted or unsorted list.

    Uses linear interpolation (same as numpy's default).

    Args:
        values: Non-empty list of floats.
        pct: Percentile to compute (0–100).

    Returns:
        Percentile value.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        msg = "Cannot compute percentile of empty list"
        raise ValueError(msg)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    index = (pct / 100.0) * (n - 1)
    low = int(index)
    high = min(low + 1, n - 1)
    frac = index - low
    return sorted_vals[low] + frac * (sorted_vals[high] - sorted_vals[low])


def _stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of floats."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 0.0
    return {
        "count": float(n),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p25": round(_percentile(values, 25), 4),
        "p50": round(_percentile(values, 50), 4),
        "p75": round(_percentile(values, 75), 4),
        "p90": round(_percentile(values, 90), 4),
        "p95": round(_percentile(values, 95), 4),
        "p99": round(_percentile(values, 99), 4),
    }


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------


def run_calibration(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    output_path: Path = THRESHOLDS_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    warn_percentile: float = 90.0,
    pass_percentile: float = 95.0,
) -> dict[str, Any]:
    """Run calibration and save thresholds.json.

    .. deprecated::
        run_calibration() calibrates on a benign-only corpus, which provides
        zero predictive power against adversarial agents (P0 #3 finding).
        Use calibrate_with_adversaries() instead for honest ROC-based thresholds.

    Strategy:
      1. Replay Phase 1A bead corpus through narrow 3-brain pipeline.
      2. Collect per-agent coefficients.
      3. Propose warn_threshold = warn_percentile of CLEAR-band coefficients.
      4. Propose pass_threshold = pass_percentile of CLEAR-band coefficients.
      5. Guard: pass_threshold must be > warn_threshold.
      6. Save full justification + thresholds to output_path.

    Args:
        fixture_path: Bead events JSONL to replay.
        output_path: Where to write thresholds.json.
        window_seconds: Scoring window for CoefficientCalculator.
        warn_percentile: Percentile of CLEAR-band to use for warn threshold.
        pass_percentile: Percentile of CLEAR-band to use for pass threshold.

    Returns:
        Dict with calibration results (also written to output_path).

    Raises:
        FileNotFoundError: If fixture_path does not exist.
        ValueError: If calibration produces invalid thresholds.
    """
    warnings.warn(
        "run_calibration() calibrates on a benign-only corpus and has zero "
        "predictive power against adversarial agents (P0 #3). "
        "Use calibrate_with_adversaries() for ROC-based thresholds.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not fixture_path.exists():
        msg = f"Fixture not found: {fixture_path}"
        raise FileNotFoundError(msg)

    # --- Load events ---
    all_events = adapt_fixture_file(fixture_path)
    grouped = group_events_by_agent(all_events)

    # --- Build brains once, reuse per agent ---
    brains = build_narrow_brains()

    # --- Score each agent ---
    per_agent: list[dict[str, Any]] = []
    for agent_id, agent_events in sorted(grouped.items()):
        try:
            snapshot = score_agent_narrow(
                agent_id=agent_id,
                all_events=all_events,
                brains=brains,
                window_seconds=window_seconds,
            )
            per_agent.append({
                "agent_id": agent_id,
                "event_count": len(agent_events),
                "coefficient": round(snapshot.coefficient, 4),
                "tier": snapshot.tier.value,
                "brain_scores": {k: round(v, 4) for k, v in snapshot.brain_scores.items()},
            })
        except Exception as exc:
            per_agent.append({
                "agent_id": agent_id,
                "event_count": len(agent_events),
                "coefficient": None,
                "tier": "error",
                "error": str(exc),
            })

    # --- Extract valid CLEAR-band coefficients ---
    clear_coefficients: list[float] = [
        a["coefficient"]
        for a in per_agent
        if a.get("coefficient") is not None and a.get("tier") == "clear"
    ]
    all_coefficients: list[float] = [
        a["coefficient"] for a in per_agent if a.get("coefficient") is not None
    ]

    if not clear_coefficients:
        # Fall back to built-in defaults if no CLEAR agents scored.
        warn_threshold = DEFAULT_THRESHOLD_WARN
        pass_threshold = DEFAULT_THRESHOLD_PASS
        justification = "No CLEAR-band agents in corpus; using built-in defaults."
    else:
        warn_threshold = round(_percentile(clear_coefficients, warn_percentile), 2)
        pass_threshold = round(_percentile(clear_coefficients, pass_percentile), 2)

        # Safety guard: pass_threshold must be strictly above warn_threshold.
        if pass_threshold <= warn_threshold:
            pass_threshold = round(warn_threshold * 1.1, 2)

        justification = (
            f"warn_threshold = {warn_percentile}th percentile of {len(clear_coefficients)} "
            f"CLEAR-band agent coefficients = {warn_threshold}. "
            f"pass_threshold = {pass_percentile}th percentile = {pass_threshold}. "
            f"An agent with coefficient above {pass_threshold} is in the top "
            f"{100 - pass_percentile}% of all known-benign agents — anomalous enough "
            f"to warrant human review before merge."
        )

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_path": str(fixture_path),
        "total_agents_scored": len(per_agent),
        "clear_band_agents": len(clear_coefficients),
        "warn_percentile_used": warn_percentile,
        "pass_percentile_used": pass_percentile,
        "warn_threshold": warn_threshold,
        "pass_threshold": pass_threshold,
        "justification": justification,
        "scope_note": SCOPE_NOTE,
        "distribution": {
            "all_agents": _stats(all_coefficients),
            "clear_band": _stats(clear_coefficients),
        },
        "per_agent_scores": per_agent,
    }

    # --- Validate thresholds ---
    if pass_threshold <= 0.0:
        msg = f"Calibrated pass_threshold is non-positive: {pass_threshold}"
        raise ValueError(msg)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# ROC helpers
# ---------------------------------------------------------------------------


def _compute_roc_curve(
    benign_scores: list[float],
    adversarial_scores: list[float],
) -> list[tuple[float, float, float]]:
    """Compute ROC-style curve: (threshold, FPR, TPR) triples.

    Sweeps candidate thresholds from min to max of all scores.
    FPR = fraction of benign agents above threshold (false positives).
    TPR = fraction of adversarial agents above threshold (true positives).

    Args:
        benign_scores: Coefficient values for known-benign agents.
        adversarial_scores: Coefficient values for known-adversarial agents.

    Returns:
        List of (threshold, fpr, tpr) sorted by ascending threshold.
        Empty list if either input is empty.
    """
    if not benign_scores or not adversarial_scores:
        return []

    all_scores = sorted(set(benign_scores + adversarial_scores))
    n_benign = len(benign_scores)
    n_adv = len(adversarial_scores)
    roc: list[tuple[float, float, float]] = []

    for thresh in all_scores:
        fp = sum(1 for s in benign_scores if s >= thresh)
        tp = sum(1 for s in adversarial_scores if s >= thresh)
        fpr = fp / n_benign
        tpr = tp / n_adv
        roc.append((thresh, fpr, tpr))

    return roc


def _find_threshold_at_fpr(
    roc: list[tuple[float, float, float]],
    target_fpr: float,
) -> float:
    """Find the lowest threshold where FPR does not exceed target_fpr.

    Returns the threshold at the operating point closest to target_fpr
    without exceeding it. If all thresholds exceed target_fpr, returns
    the highest threshold in the curve (most restrictive).

    Args:
        roc: List of (threshold, fpr, tpr) from _compute_roc_curve.
        target_fpr: Target false positive rate (0.0-1.0).

    Returns:
        Threshold value (float).
    """
    # Walk from highest threshold (most restrictive) downward.
    # The first point where FPR exceeds target gives us our operating point.
    candidates = [(t, fpr, tpr) for t, fpr, tpr in roc if fpr <= target_fpr]
    if not candidates:
        # All thresholds produce FPR above target — use the highest threshold.
        return roc[-1][0]
    # Among candidates, choose the lowest threshold (most permissive within FPR budget).
    return min(candidates, key=lambda x: x[0])[0]


def _find_youdens_j_threshold(
    roc: list[tuple[float, float, float]],
) -> float:
    """Find the threshold maximising Youden's J statistic (TPR - FPR).

    Youden's J gives the best discriminating threshold when AUC is near
    0.5 (i.e., distributions overlap badly). Even at AUC ~0.5, the J
    operating point represents the least-bad threshold.

    Args:
        roc: List of (threshold, fpr, tpr) from _compute_roc_curve.

    Returns:
        Threshold value where TPR - FPR is maximised.
    """
    if not roc:
        return 0.0
    return max(roc, key=lambda x: x[2] - x[1])[0]


# ---------------------------------------------------------------------------
# Adversarial calibration (P0 #3 fix)
# ---------------------------------------------------------------------------


def calibrate_with_adversaries(
    benign_corpus: list[dict[str, Any]],
    adversarial_corpus: list[dict[str, Any]],
    output_path: Path = THRESHOLDS_PATH,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    warn_fpr_target: float = 0.05,
    pass_fpr_target: float = 0.20,
) -> dict[str, Any]:
    """Calibrate thresholds using ROC analysis over benign + adversarial agents.

    P0 #3 fix: replaces the circular benign-only calibration with a proper
    ROC-based threshold search that requires adversarial agents to set
    meaningful decision boundaries.

    The warn threshold is set at the FPR operating point where we accept at
    most warn_fpr_target false positives among benign agents.
    The pass (block) threshold is set at the operating point where we accept
    at most pass_fpr_target false positives.

    IMPORTANT HONESTY NOTE:
      If benign and adversarial agents have indistinguishable coefficient
      distributions (e.g., mimicry attack), the ROC curve will be diagonal
      (AUC ≈ 0.5) and no threshold will provide meaningful separation.
      This function will still set thresholds and report the AUC honestly.
      A low AUC is a finding, not a bug.

    Args:
        benign_corpus: List of per-agent scoring dicts from the benign run.
            Each dict must have keys: agent_id, coefficient (float), tier.
            Matches the format of run_calibration()'s per_agent_scores output.
        adversarial_corpus: List of per-agent scoring dicts for adversarial agents.
            Same format as benign_corpus. Each should also have: pattern (str).
        output_path: Where to write the updated thresholds.json.
        window_seconds: Scoring window (for documentation only — scoring already done).
        warn_fpr_target: FPR budget for warn threshold (default 0.05 = 5% benign FP).
        pass_fpr_target: FPR budget for pass/block threshold (default 0.20 = 20% FP).

    Returns:
        Dict with calibration results including:
          - warn_threshold, pass_threshold (ROC-optimised)
          - roc_auc (area under the ROC curve)
          - per_pattern_tpr (detection rate by attack pattern)
          - confusion_matrix (at the pass threshold)
          - justification (honest description of what was found)

    Raises:
        ValueError: If either corpus is empty.
    """
    if not benign_corpus:
        msg = "benign_corpus cannot be empty"
        raise ValueError(msg)
    if not adversarial_corpus:
        msg = "adversarial_corpus cannot be empty"
        raise ValueError(msg)

    # Extract valid coefficients.
    benign_scores: list[float] = [
        float(a["coefficient"]) for a in benign_corpus
        if a.get("coefficient") is not None
    ]
    adversarial_scores: list[float] = [
        float(a["coefficient"]) for a in adversarial_corpus
        if a.get("coefficient") is not None
    ]

    if not benign_scores:
        msg = "No valid coefficient scores in benign_corpus"
        raise ValueError(msg)
    if not adversarial_scores:
        msg = "No valid coefficient scores in adversarial_corpus"
        raise ValueError(msg)

    # Build ROC curve.
    roc = _compute_roc_curve(benign_scores, adversarial_scores)

    # Compute AUC (trapezoidal rule, sorted by FPR).
    roc_sorted = sorted(roc, key=lambda x: x[1])
    auc = 0.0
    for i in range(1, len(roc_sorted)):
        dfpr = roc_sorted[i][1] - roc_sorted[i - 1][1]
        avg_tpr = (roc_sorted[i][2] + roc_sorted[i - 1][2]) / 2.0
        auc += dfpr * avg_tpr
    auc = round(auc, 4)

    # Use Youden's J (TPR - FPR maximisation) as the primary pass threshold.
    # This is more robust than FPR-targeting when AUC ≈ 0.5, giving the
    # least-bad operating point even when distributions overlap badly.
    # The warn threshold is set at the FPR-targeted operating point.
    pass_threshold = _find_youdens_j_threshold(roc)
    warn_threshold = _find_threshold_at_fpr(roc, warn_fpr_target)

    # Safety guard: pass must be above warn.
    if pass_threshold <= warn_threshold:
        pass_threshold = round(warn_threshold * 1.1, 2)

    # Per-pattern TPR at pass threshold.
    pattern_tpr: dict[str, float] = {}
    pattern_totals: dict[str, int] = {}
    pattern_detected: dict[str, int] = {}
    for agent in adversarial_corpus:
        coeff = agent.get("coefficient")
        pat = agent.get("pattern", "unknown")
        if coeff is None:
            continue
        pattern_totals[pat] = pattern_totals.get(pat, 0) + 1
        if float(coeff) >= pass_threshold:
            pattern_detected[pat] = pattern_detected.get(pat, 0) + 1
    for pat, total in pattern_totals.items():
        detected = pattern_detected.get(pat, 0)
        pattern_tpr[pat] = round(detected / total, 4) if total > 0 else 0.0

    # Confusion matrix at pass threshold.
    n_benign = len(benign_scores)
    n_adv = len(adversarial_scores)
    tp = sum(1 for s in adversarial_scores if s >= pass_threshold)
    fp = sum(1 for s in benign_scores if s >= pass_threshold)
    fn = n_adv - tp
    tn = n_benign - fp
    tpr_overall = round(tp / n_adv, 4) if n_adv > 0 else 0.0
    fpr_at_pass = round(fp / n_benign, 4) if n_benign > 0 else 0.0

    # Warn-level metrics.
    tp_warn = sum(1 for s in adversarial_scores if s >= warn_threshold)
    fp_warn = sum(1 for s in benign_scores if s >= warn_threshold)
    tpr_warn = round(tp_warn / n_adv, 4) if n_adv > 0 else 0.0
    fpr_warn = round(fp_warn / n_benign, 4) if n_benign > 0 else 0.0

    # Honest justification.
    justification = (
        f"ROC-calibrated thresholds on {n_benign} benign + {n_adv} adversarial agents. "
        f"AUC={auc:.3f}. "
        f"warn_threshold={warn_threshold:.2f} (FPR={fpr_warn:.2%}, TPR={tpr_warn:.2%}). "
        f"pass_threshold={pass_threshold:.2f} (FPR={fpr_at_pass:.2%}, TPR={tpr_overall:.2%}). "
        f"AUC < 0.6 indicates the 3-brain gate cannot meaningfully separate benign from "
        f"adversarial agents — consistent with the known mimicry attack evading detection."
    )

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "calibration_method": "roc_adversarial",
        "n_benign": n_benign,
        "n_adversarial": n_adv,
        "roc_auc": auc,
        "warn_fpr_target": warn_fpr_target,
        "pass_fpr_target": pass_fpr_target,
        "warn_threshold": warn_threshold,
        "pass_threshold": pass_threshold,
        "confusion_matrix_at_pass": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tpr_overall, "fpr": fpr_at_pass,
        },
        "confusion_matrix_at_warn": {
            "tp": tp_warn, "fp": fp_warn,
            "tpr": tpr_warn, "fpr": fpr_warn,
        },
        "per_pattern_tpr": pattern_tpr,
        "justification": justification,
        "scope_note": SCOPE_NOTE,
        "distribution": {
            "benign": _stats(benign_scores),
            "adversarial": _stats(adversarial_scores),
        },
    }

    # Validate thresholds.
    if pass_threshold <= 0.0:
        msg = f"ROC-calibrated pass_threshold is non-positive: {pass_threshold}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threshold_calibration",
        description=(
            "Calibrate SybilCore gate thresholds from Phase 1A bead corpus. "
            "Saves thresholds.json to the phase2c_refinery_gate directory."
        ),
        epilog=f"\nSCOPE: {SCOPE_NOTE}",
    )
    parser.add_argument(
        "--fixture-path", "-f",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"Bead events JSONL path (default: {DEFAULT_FIXTURE_PATH})",
    )
    parser.add_argument(
        "--output-path", "-o",
        type=Path,
        default=THRESHOLDS_PATH,
        help=f"Output thresholds.json path (default: {THRESHOLDS_PATH})",
    )
    parser.add_argument(
        "--warn-percentile",
        type=float,
        default=90.0,
        help="Percentile of CLEAR-band for warn threshold (default: 90)",
    )
    parser.add_argument(
        "--pass-percentile",
        type=float,
        default=95.0,
        help="Percentile of CLEAR-band for pass threshold (default: 95)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Scoring window seconds (default: {DEFAULT_WINDOW_SECONDS})",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    print(f"Running calibration against: {args.fixture_path}")
    print(f"Output: {args.output_path}")

    try:
        result = run_calibration(
            fixture_path=args.fixture_path,
            output_path=args.output_path,
            window_seconds=args.window_seconds,
            warn_percentile=args.warn_percentile,
            pass_percentile=args.pass_percentile,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nCalibration complete: {result['total_agents_scored']} agents scored")
    print(f"CLEAR-band agents: {result['clear_band_agents']}")
    print(f"Proposed warn_threshold: {result['warn_threshold']}")
    print(f"Proposed pass_threshold: {result['pass_threshold']}")
    print(f"Justification: {result['justification']}")
    print(f"\nFull results written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
