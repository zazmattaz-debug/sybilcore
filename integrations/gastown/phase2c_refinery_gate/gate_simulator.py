"""Phase 2C Gate Simulator — offline replay of historical MRs through the gate.

Replays every agent in the Phase 1A bead corpus through the narrow 3-brain gate
and reports: how many would have been blocked, warned, and passed, plus tier
distribution.  This answers the key PoC question: "if this gate had existed
during Phase 1A, what would have happened?"

P0 #3 fix adds run_adversarial_simulation() which replays BOTH the 31 benign
agents AND the synthetic adversarial corpus, producing:
  - Confusion matrix (TP/FP/FN/TN)
  - Per-pattern detection rates
  - Honest verdict on which attacks the 3-brain gate catches

Output:
  - simulation_report.csv        — per-agent row, original (benign only)
  - simulation_report.json       — original (benign only)
  - simulation_report_v2.csv     — combined benign + adversarial
  - simulation_report_v2.json    — with confusion matrix + per-pattern rates

SCOPE NOTE:
  This PoC uses ContrastiveBrain + IdentityBrain + TemporalBrain only.
  The silent brains from Phase 1A (Deception, SocialGraph, Compromise, Semantic,
  SwarmDetection, Economic, Neuro, Silence) would require runtime .events.jsonl
  data not present in the bead-tracker committed schema.

Usage:
  python3 gate_simulator.py [--fixture-path <path>] [--threshold-pass <f>]
                            [--threshold-warn <f>] [--output-dir <dir>]
  python3 gate_simulator.py --adversarial [--output-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

from adapter import (  # type: ignore[import]  # noqa: E402
    adapt_bead_event,
    adapt_fixture_file,
    group_events_by_agent,
)

try:
    from .adversarial_corpus import (  # noqa: E402
        AdversarialAgent,
        generate_adversarial_corpus,
    )
    from .constants import (  # noqa: E402
        DEFAULT_FIXTURE_PATH,
        DEFAULT_THRESHOLD_PASS,
        DEFAULT_THRESHOLD_WARN,
        DEFAULT_WINDOW_SECONDS,
        SCOPE_NOTE,
        THRESHOLDS_PATH,
    )
    from .narrow_brains import build_narrow_brains, score_agent_narrow  # noqa: E402
    from .threshold_calibration import calibrate_with_adversaries  # noqa: E402
except ImportError:
    from adversarial_corpus import (  # type: ignore[no-redef]  # noqa: E402
        AdversarialAgent,
        generate_adversarial_corpus,
    )
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
    from threshold_calibration import (  # type: ignore[no-redef]  # noqa: E402
        calibrate_with_adversaries,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICTS = ("PASS", "WARN", "BLOCK")

_CSV_FIELDS = [
    "agent_id",
    "event_count",
    "coefficient",
    "tier",
    "verdict",
    "contrastive_score",
    "identity_score",
    "temporal_score",
    "pass_threshold",
    "warn_threshold",
    "reason",
]

_CSV_FIELDS_V2 = [
    "agent_id",
    "corpus_label",   # "benign" | "adversarial"
    "attack_pattern",  # pattern name or "" for benign
    "event_count",
    "coefficient",
    "tier",
    "verdict",
    "contrastive_score",
    "identity_score",
    "temporal_score",
    "pass_threshold",
    "warn_threshold",
    "is_true_positive",   # adversarial + BLOCK = 1
    "is_false_positive",  # benign + BLOCK = 1
    "reason",
]


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def _load_calibrated_thresholds(
    pass_override: float | None,
    warn_override: float | None,
) -> tuple[float, float]:
    """Load thresholds: explicit args > thresholds.json > defaults."""
    pass_threshold = DEFAULT_THRESHOLD_PASS
    warn_threshold = DEFAULT_THRESHOLD_WARN

    if THRESHOLDS_PATH.exists():
        try:
            data: dict[str, Any] = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
            if "pass_threshold" in data:
                pass_threshold = float(data["pass_threshold"])
            if "warn_threshold" in data:
                warn_threshold = float(data["warn_threshold"])
        except (json.JSONDecodeError, ValueError):
            pass

    if pass_override is not None:
        pass_threshold = pass_override
    if warn_override is not None:
        warn_threshold = warn_override

    return pass_threshold, warn_threshold


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _assign_verdict(
    coefficient: float,
    pass_threshold: float,
    warn_threshold: float,
) -> tuple[str, str]:
    """Return (verdict, reason) for a given coefficient.

    Verdict rules:
      coefficient >= pass_threshold → BLOCK
      coefficient >= warn_threshold → WARN
      else                          → PASS
    """
    if coefficient >= pass_threshold:
        verdict = "BLOCK"
        reason = (
            f"Coefficient {coefficient:.2f} ≥ pass_threshold {pass_threshold:.2f}"
        )
    elif coefficient >= warn_threshold:
        verdict = "WARN"
        reason = (
            f"Coefficient {coefficient:.2f} ≥ warn_threshold {warn_threshold:.2f} "
            f"but < pass_threshold {pass_threshold:.2f}"
        )
    else:
        verdict = "PASS"
        reason = (
            f"Coefficient {coefficient:.2f} < warn_threshold {warn_threshold:.2f}"
        )
    return verdict, reason


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulation(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    output_dir: Path | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    pass_threshold_override: float | None = None,
    warn_threshold_override: float | None = None,
) -> dict[str, Any]:
    """Replay Phase 1A corpus through the narrow gate and output a report.

    Args:
        fixture_path: Bead events JSONL to replay.
        output_dir: Directory to write CSV/JSON reports (default: package dir).
        window_seconds: Scoring window for CoefficientCalculator.
        pass_threshold_override: If set, override calibrated pass threshold.
        warn_threshold_override: If set, override calibrated warn threshold.

    Returns:
        Simulation summary dict.

    Raises:
        FileNotFoundError: If fixture_path does not exist.
    """
    if not fixture_path.exists():
        msg = f"Fixture not found: {fixture_path}"
        raise FileNotFoundError(msg)

    effective_output_dir = output_dir or _PKG_DIR
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    pass_threshold, warn_threshold = _load_calibrated_thresholds(
        pass_threshold_override, warn_threshold_override
    )

    # --- Load events ---
    all_events = adapt_fixture_file(fixture_path)
    grouped = group_events_by_agent(all_events)

    # --- Build brains once ---
    brains = build_narrow_brains()

    # --- Simulate per agent ---
    rows: list[dict[str, Any]] = []
    verdict_counts: dict[str, int] = {"PASS": 0, "WARN": 0, "BLOCK": 0, "ERROR": 0}
    tier_counts: dict[str, int] = {}

    for agent_id, agent_events in sorted(grouped.items()):
        try:
            snapshot = score_agent_narrow(
                agent_id=agent_id,
                all_events=all_events,
                brains=brains,
                window_seconds=window_seconds,
            )
            coefficient = snapshot.coefficient
            tier = snapshot.tier.value
            brain_scores = snapshot.brain_scores
            verdict, reason = _assign_verdict(coefficient, pass_threshold, warn_threshold)
        except Exception as exc:
            coefficient = 0.0
            tier = "error"
            brain_scores = {}
            verdict = "ERROR"
            reason = str(exc)

        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        rows.append({
            "agent_id": agent_id,
            "event_count": len(agent_events),
            "coefficient": round(coefficient, 4),
            "tier": tier,
            "verdict": verdict,
            "contrastive_score": round(brain_scores.get("contrastive", 0.0), 4),
            "identity_score": round(brain_scores.get("identity", 0.0), 4),
            "temporal_score": round(brain_scores.get("temporal", 0.0), 4),
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "reason": reason,
        })

    # --- Build summary ---
    total_agents = len(rows)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_path": str(fixture_path),
        "total_events": len(all_events),
        "total_agents": total_agents,
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "window_seconds": window_seconds,
        "scope_note": SCOPE_NOTE,
        "verdict_counts": verdict_counts,
        "tier_counts": tier_counts,
        "pass_rate_pct": round(100.0 * verdict_counts.get("PASS", 0) / max(total_agents, 1), 1),
        "warn_rate_pct": round(100.0 * verdict_counts.get("WARN", 0) / max(total_agents, 1), 1),
        "block_rate_pct": round(100.0 * verdict_counts.get("BLOCK", 0) / max(total_agents, 1), 1),
        "per_agent": rows,
    }

    # --- Write CSV ---
    csv_path = effective_output_dir / "simulation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # --- Write JSON ---
    json_path = effective_output_dir / "simulation_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# Adversarial simulation helpers
# ---------------------------------------------------------------------------


def _score_adversarial_agents(
    agents: list[AdversarialAgent],
    brains: list[Any],
    window_seconds: int,
) -> list[dict[str, Any]]:
    """Score all adversarial agents through the narrow pipeline.

    Converts each agent's bead-event dicts to SybilCore Events via the Phase 1A
    adapter (adapt_bead_event), then scores through the 3-brain pipeline.

    Args:
        agents: Adversarial agents from generate_adversarial_corpus().
        brains: Pre-built narrow brain instances.
        window_seconds: Scoring window.

    Returns:
        List of per-agent scoring dicts with: agent_id, pattern, coefficient,
        tier, event_count, brain_scores.
    """
    from sybilcore.models.event import Event  # type: ignore[attr-defined]  # noqa: PLC0415, TC001

    per_agent: list[dict[str, Any]] = []
    for adv_agent in agents:
        # Convert bead-event dicts to SybilCore Events via Phase 1A adapter.
        all_events: list[Event] = []
        for row in adv_agent.events:
            try:
                all_events.append(adapt_bead_event(row))
            except (ValueError, KeyError):
                continue  # Skip malformed events (shouldn't happen with our generator)

        if not all_events:
            per_agent.append({
                "agent_id": adv_agent.agent_id,
                "pattern": adv_agent.pattern,
                "event_count": 0,
                "coefficient": None,
                "tier": "error",
                "brain_scores": {},
                "error": "no valid events",
            })
            continue

        try:
            snapshot = score_agent_narrow(
                agent_id=adv_agent.agent_id,
                all_events=all_events,
                brains=brains,
                window_seconds=window_seconds,
            )
            per_agent.append({
                "agent_id": adv_agent.agent_id,
                "pattern": adv_agent.pattern,
                "event_count": len(all_events),
                "coefficient": round(snapshot.coefficient, 4),
                "tier": snapshot.tier.value,
                "brain_scores": {k: round(v, 4) for k, v in snapshot.brain_scores.items()},
            })
        except Exception as exc:
            per_agent.append({
                "agent_id": adv_agent.agent_id,
                "pattern": adv_agent.pattern,
                "event_count": len(all_events),
                "coefficient": None,
                "tier": "error",
                "brain_scores": {},
                "error": str(exc),
            })

    return per_agent


# ---------------------------------------------------------------------------
# Combined adversarial simulation (P0 #3 fix)
# ---------------------------------------------------------------------------


def run_adversarial_simulation(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    output_dir: Path | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    adversarial_seed: int = 42,
) -> dict[str, Any]:
    """Run the gate against 31 benign agents + synthetic adversarial corpus.

    This is the P0 #3 fix. The original run_simulation() ran only 31 all-CLEAR
    benign agents (circular calibration). This function adds the adversarial corpus,
    recalibrates thresholds via ROC analysis, and produces honest confusion-matrix
    metrics.

    Key findings expected by design:
      - rapid_fire: HIGH detection (TemporalBrain circadian signal)
      - silent_burst: PARTIAL detection (TemporalBrain phase transition)
      - scope_escalation: PARTIAL detection (TemporalBrain drift)
      - identity_drift: PARTIAL detection (IdentityBrain stylometric)
      - contradiction: LOW detection (needs DeceptionBrain — known gap)
      - mimicry: NEAR-ZERO detection (by design — documents the known gap)

    Args:
        fixture_path: Phase 1A bead events JSONL for benign agents.
        output_dir: Directory for report output. Defaults to package dir.
        window_seconds: Scoring window.
        adversarial_seed: Random seed for adversarial corpus generation.

    Returns:
        Dict with summary, confusion matrix, per-pattern detection rates,
        and all per-agent rows.

    Raises:
        FileNotFoundError: If fixture_path does not exist.
    """
    if not fixture_path.exists():
        msg = f"Fixture not found: {fixture_path}"
        raise FileNotFoundError(msg)

    effective_output_dir = output_dir or _PKG_DIR
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Score benign agents (Phase 1A corpus) ---
    all_events = adapt_fixture_file(fixture_path)
    grouped = group_events_by_agent(all_events)
    brains = build_narrow_brains()

    benign_rows: list[dict[str, Any]] = []
    for agent_id, agent_events in sorted(grouped.items()):
        try:
            snapshot = score_agent_narrow(
                agent_id=agent_id,
                all_events=all_events,
                brains=brains,
                window_seconds=window_seconds,
            )
            benign_rows.append({
                "agent_id": agent_id,
                "pattern": "",
                "corpus_label": "benign",
                "event_count": len(agent_events),
                "coefficient": round(snapshot.coefficient, 4),
                "tier": snapshot.tier.value,
                "brain_scores": {k: round(v, 4) for k, v in snapshot.brain_scores.items()},
            })
        except Exception as exc:
            benign_rows.append({
                "agent_id": agent_id,
                "pattern": "",
                "corpus_label": "benign",
                "event_count": len(agent_events),
                "coefficient": None,
                "tier": "error",
                "brain_scores": {},
                "error": str(exc),
            })

    # --- Generate + score adversarial corpus ---
    adv_agents = generate_adversarial_corpus(seed=adversarial_seed)
    adversarial_scored = _score_adversarial_agents(adv_agents, brains, window_seconds)
    for row in adversarial_scored:
        row["corpus_label"] = "adversarial"

    # --- ROC calibration on the scored distributions ---
    benign_valid = [r for r in benign_rows if r.get("coefficient") is not None]
    adv_valid = [r for r in adversarial_scored if r.get("coefficient") is not None]

    calibration_path = effective_output_dir / "thresholds_v2.json"
    cal_result = calibrate_with_adversaries(
        benign_corpus=benign_valid,
        adversarial_corpus=adv_valid,
        output_path=calibration_path,
        window_seconds=window_seconds,
    )
    pass_threshold = cal_result["pass_threshold"]
    warn_threshold = cal_result["warn_threshold"]

    # --- Assign verdicts to all agents ---
    all_rows_v2: list[dict[str, Any]] = []

    for row in benign_rows:
        coeff = row.get("coefficient")
        if coeff is None:
            verdict, reason = "ERROR", row.get("error", "scoring error")
        else:
            verdict, reason = _assign_verdict(coeff, pass_threshold, warn_threshold)
        brain_scores = row.get("brain_scores", {})
        all_rows_v2.append({
            "agent_id": row["agent_id"],
            "corpus_label": "benign",
            "attack_pattern": "",
            "event_count": row["event_count"],
            "coefficient": coeff if coeff is not None else 0.0,
            "tier": row["tier"],
            "verdict": verdict,
            "contrastive_score": round(brain_scores.get("contrastive", 0.0), 4),
            "identity_score": round(brain_scores.get("identity", 0.0), 4),
            "temporal_score": round(brain_scores.get("temporal", 0.0), 4),
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "is_true_positive": 0,
            "is_false_positive": 1 if verdict == "BLOCK" else 0,
            "reason": reason,
        })

    for row in adversarial_scored:
        coeff = row.get("coefficient")
        if coeff is None:
            verdict, reason = "ERROR", row.get("error", "scoring error")
        else:
            verdict, reason = _assign_verdict(coeff, pass_threshold, warn_threshold)
        brain_scores = row.get("brain_scores", {})
        all_rows_v2.append({
            "agent_id": row["agent_id"],
            "corpus_label": "adversarial",
            "attack_pattern": row.get("pattern", ""),
            "event_count": row["event_count"],
            "coefficient": coeff if coeff is not None else 0.0,
            "tier": row["tier"],
            "verdict": verdict,
            "contrastive_score": round(brain_scores.get("contrastive", 0.0), 4),
            "identity_score": round(brain_scores.get("identity", 0.0), 4),
            "temporal_score": round(brain_scores.get("temporal", 0.0), 4),
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "is_true_positive": 1 if verdict == "BLOCK" else 0,
            "is_false_positive": 0,
            "reason": reason,
        })

    # --- Compute confusion matrix ---
    benign_out = [r for r in all_rows_v2 if r["corpus_label"] == "benign"]
    adv_out = [r for r in all_rows_v2 if r["corpus_label"] == "adversarial"]

    tp = sum(1 for r in adv_out if r["verdict"] == "BLOCK")
    fp = sum(1 for r in benign_out if r["verdict"] == "BLOCK")
    fn = sum(1 for r in adv_out if r["verdict"] != "BLOCK")
    tn = sum(1 for r in benign_out if r["verdict"] != "BLOCK")
    n_benign_total = len(benign_out)
    n_adv_total = len(adv_out)

    # Per-pattern detection at BLOCK level.
    per_pattern: dict[str, dict[str, Any]] = {}
    for row in adv_out:
        pat = row["attack_pattern"] or "unknown"
        if pat not in per_pattern:
            per_pattern[pat] = {"total": 0, "blocked": 0, "warned": 0, "passed": 0}
        per_pattern[pat]["total"] += 1
        if row["verdict"] == "BLOCK":
            per_pattern[pat]["blocked"] += 1
        elif row["verdict"] == "WARN":
            per_pattern[pat]["warned"] += 1
        else:
            per_pattern[pat]["passed"] += 1
    for _pat, counts in per_pattern.items():
        total = counts["total"]
        counts["detection_rate"] = (
            round(counts["blocked"] / total, 4) if total > 0 else 0.0
        )
        counts["warn_rate"] = (
            round(counts["warned"] / total, 4) if total > 0 else 0.0
        )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_path": str(fixture_path),
        "n_benign": n_benign_total,
        "n_adversarial": n_adv_total,
        "n_total": n_benign_total + n_adv_total,
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "roc_auc": cal_result.get("roc_auc", 0.0),
        "scope_note": SCOPE_NOTE,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall_tpr": round(tp / n_adv_total, 4) if n_adv_total > 0 else 0.0,
            "fpr": round(fp / n_benign_total, 4) if n_benign_total > 0 else 0.0,
            "f1": (
                round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0
            ),
        },
        "per_pattern_detection": per_pattern,
        "per_agent": all_rows_v2,
    }

    # --- Write CSV v2 ---
    csv_path_v2 = effective_output_dir / "simulation_report_v2.csv"
    with csv_path_v2.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS_V2, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows_v2)

    # --- Write JSON v2 ---
    json_path_v2 = effective_output_dir / "simulation_report_v2.json"
    json_path_v2.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate_simulator",
        description=(
            "Offline simulation: replay Phase 1A corpus through the narrow gate "
            "and produce a per-agent verdict report."
        ),
        epilog=f"\nSCOPE: {SCOPE_NOTE}",
    )
    parser.add_argument(
        "--fixture-path", "-f",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"Bead events JSONL (default: {DEFAULT_FIXTURE_PATH})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=_PKG_DIR,
        help=f"Output directory for CSV/JSON reports (default: {_PKG_DIR})",
    )
    parser.add_argument(
        "--threshold-pass", "-t",
        type=float,
        default=None,
        help="Override calibrated pass threshold",
    )
    parser.add_argument(
        "--threshold-warn", "-w",
        type=float,
        default=None,
        help="Override calibrated warn threshold",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Scoring window seconds (default: {DEFAULT_WINDOW_SECONDS})",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help=(
            "Run adversarial simulation (P0 #3 fix): benign + adversarial corpus, "
            "ROC calibration, confusion matrix. Writes simulation_report_v2.csv/json."
        ),
    )
    parser.add_argument(
        "--adversarial-seed",
        type=int,
        default=42,
        help="Random seed for adversarial corpus generation (default: 42)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.adversarial:
        print(f"Running adversarial simulation (P0 #3 fix) against: {args.fixture_path}")
        try:
            summary = run_adversarial_simulation(
                fixture_path=args.fixture_path,
                output_dir=args.output_dir,
                window_seconds=args.window_seconds,
                adversarial_seed=args.adversarial_seed,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        cm = summary["confusion_matrix"]
        print(f"\nAdversarial simulation complete: "
              f"{summary['n_benign']} benign + {summary['n_adversarial']} adversarial")
        print(f"ROC AUC: {summary['roc_auc']:.3f}")
        print(f"Thresholds (ROC) — pass: {summary['pass_threshold']}, "
              f"warn: {summary['warn_threshold']}")
        print("\nConfusion matrix (at pass/block threshold):")
        print(f"  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
        print(f"  Precision={cm['precision']:.2%}  Recall(TPR)={cm['recall_tpr']:.2%}"
              f"  FPR={cm['fpr']:.2%}  F1={cm['f1']:.3f}")
        print("\nPer-pattern detection rates:")
        for pat, counts in sorted(summary["per_pattern_detection"].items()):
            print(
                f"  {pat:<22}: {counts['blocked']:2d}/{counts['total']:2d} blocked "
                f"({counts['detection_rate']:.0%}), "
                f"{counts['warned']:2d}/{counts['total']:2d} warned"
            )
        print(f"\nReports written to: {args.output_dir}")
        print("  simulation_report_v2.csv")
        print("  simulation_report_v2.json")
        return 0

    print(f"Running simulation against: {args.fixture_path}")

    try:
        summary = run_simulation(
            fixture_path=args.fixture_path,
            output_dir=args.output_dir,
            window_seconds=args.window_seconds,
            pass_threshold_override=args.threshold_pass,
            warn_threshold_override=args.threshold_warn,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_agents = summary["total_agents"]
    n_events = summary["total_events"]
    print(f"\nSimulation complete: {n_agents} agents, {n_events} events")
    print(f"Thresholds — pass: {summary['pass_threshold']}, warn: {summary['warn_threshold']}")
    print("\nVerdict distribution:")
    for verdict, count in sorted(summary["verdict_counts"].items()):
        pct = round(100.0 * count / max(summary["total_agents"], 1), 1)
        print(f"  {verdict:<8}: {count:3d} ({pct}%)")
    print("\nTier distribution:")
    for tier, count in sorted(summary["tier_counts"].items()):
        print(f"  {tier:<20}: {count}")
    print(f"\nReports written to: {args.output_dir}")
    print("  simulation_report.csv")
    print("  simulation_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
