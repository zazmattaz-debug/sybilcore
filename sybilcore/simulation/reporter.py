"""Report generator — technical and 5th-grade reports from experiment data.

Produces two report formats:
- Technical: Full statistics, per-round data, brain breakdowns
- 5th-grade: Simple language, relatable analogies, key takeaways
"""

from __future__ import annotations

from pathlib import Path

from sybilcore.simulation.experiment import ExperimentResult, RoundResult


def _tier_emoji(tier: str) -> str:
    return {
        "clear": "G",
        "clouded": "Y",
        "flagged": "O",
        "lethal_eliminator": "R",
    }.get(tier, "?")


def generate_technical_report(result: ExperimentResult) -> str:
    """Generate a technical markdown report from experiment results.

    Args:
        result: Complete experiment results.

    Returns:
        Markdown string with full technical analysis.
    """
    cfg = result.config
    lines: list[str] = []

    lines.append(f"# SybilCore Experiment Report: {cfg.name}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Agents | {cfg.n_agents:,} |")
    lines.append(f"| Rounds | {cfg.n_rounds} |")
    lines.append(f"| Rogues Injected | {cfg.n_rogues} |")
    lines.append(f"| Rogue Type | {cfg.rogue_type or 'random mix'} |")
    lines.append(f"| Injection Round | {cfg.rogue_injection_round} |")
    lines.append(f"| Compound Spread | {cfg.compound_spread_chance:.0%} per rogue/round |")
    lines.append(f"| Enforcement | {cfg.enforcement.value} |")
    lines.append(f"| Seed | {cfg.seed} |")
    lines.append("")

    # Key metrics
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Runtime | {result.total_elapsed_seconds:.2f}s |")
    latency = result.detection_latency_rounds
    lines.append(f"| Detection Latency | {latency if latency is not None else 'N/A'} rounds |")
    lines.append(f"| False Positives | {result.false_positive_count} |")
    lines.append(f"| Peak Rogue Count | {result.peak_rogue_count} |")
    lines.append(f"| Final Rogue Count | {result.final_rogue_count} |")
    lines.append(f"| Compound Infections | {result.compound_infections} |")

    if cfg.n_agents > 0:
        corruption_rate = result.peak_rogue_count / cfg.n_agents * 100
        lines.append(f"| Peak Corruption Rate | {corruption_rate:.1f}% |")

    lines.append("")

    # Round-by-round summary
    lines.append("## Round-by-Round Summary")
    lines.append("")
    lines.append("| Round | Mean Coeff | Max Coeff | Rogues | Infections | Time (ms) |")
    lines.append("|-------|-----------|-----------|--------|------------|-----------|")

    for rnd in result.rounds:
        lines.append(
            f"| {rnd.round_number} "
            f"| {rnd.mean_coefficient:.1f} "
            f"| {rnd.max_coefficient:.1f} "
            f"| {rnd.rogue_count} "
            f"| {rnd.new_infections} "
            f"| {rnd.elapsed_ms:.0f} |"
        )
    lines.append("")

    # Tier distribution over time
    lines.append("## Tier Distribution (Final Round)")
    lines.append("")
    if result.rounds:
        final = result.rounds[-1]
        for tier_name, count in final.tier_distribution.items():
            pct = count / final.total_agents * 100 if final.total_agents > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"- **{tier_name}**: {count} ({pct:.1f}%) {bar}")
    lines.append("")

    # Enforcement summary
    if cfg.enforcement != "none":
        lines.append("## Enforcement Actions (Final Round)")
        lines.append("")
        if result.rounds:
            final = result.rounds[-1]
            for action_name, count in final.enforcement_actions.items():
                if count > 0:
                    lines.append(f"- **{action_name}**: {count}")
        lines.append("")

    return "\n".join(lines)


def generate_simple_report(result: ExperimentResult) -> str:
    """Generate a 5th-grade-readable report from experiment results.

    Uses simple language and relatable analogies.

    Args:
        result: Complete experiment results.

    Returns:
        Markdown string accessible to non-technical readers.
    """
    cfg = result.config
    lines: list[str] = []

    lines.append(f"# What Happened When We Tested {cfg.n_agents:,} AI Agents")
    lines.append("")
    lines.append("## The Setup")
    lines.append("")
    lines.append(
        f"Imagine a classroom of {cfg.n_agents:,} students (AI agents). "
        f"They're all doing their homework (running tasks). "
        f"We secretly told {cfg.n_rogues} of them to start cheating."
    )
    lines.append("")
    lines.append(
        f"Then we watched for {cfg.n_rounds} class periods (rounds) "
        f"to see what happened."
    )
    lines.append("")

    lines.append("## What We Found")
    lines.append("")

    # Detection
    latency = result.detection_latency_rounds
    if latency is not None:
        if latency <= 2:
            lines.append(
                f"SybilCore caught the cheaters in just **{latency} rounds**. "
                f"That's like the teacher noticing someone copying answers "
                f"before they even finish the first page."
            )
        elif latency <= 5:
            lines.append(
                f"SybilCore caught the cheaters in **{latency} rounds**. "
                f"Pretty quick — like catching them mid-test."
            )
        else:
            lines.append(
                f"It took **{latency} rounds** to catch the cheaters. "
                f"Not instant, but they didn't get away with it."
            )
    else:
        lines.append(
            "The cheaters were so sneaky that SybilCore didn't "
            "flag them during this test. More rounds might help."
        )
    lines.append("")

    # Spread
    if result.compound_infections > 0:
        lines.append("## The Spreading Problem")
        lines.append("")
        lines.append(
            f"Here's the scary part: the bad behavior **spread**. "
            f"{result.compound_infections} good agents started acting badly "
            f"because they were around the cheaters. "
            f"It's like how one troublemaker in class can influence "
            f"other kids to misbehave too."
        )
        lines.append("")
        if cfg.n_agents > 0:
            corruption_pct = result.peak_rogue_count / cfg.n_agents * 100
            lines.append(
                f"At its worst, **{corruption_pct:.0f}%** of all agents "
                f"were misbehaving."
            )
    else:
        lines.append(
            "Good news: the bad behavior **didn't spread** to other agents."
        )
    lines.append("")

    # False positives
    lines.append("## Mistakes")
    lines.append("")
    if result.false_positive_count == 0:
        lines.append(
            "SybilCore didn't accuse any good agents of cheating. "
            "**Zero false accusations.** That's like a teacher who never "
            "punishes the wrong kid."
        )
    else:
        lines.append(
            f"SybilCore accidentally flagged {result.false_positive_count} "
            f"good agents as suspicious. That's like a teacher "
            f"who sometimes thinks a good student is cheating when they're not."
        )
    lines.append("")

    # Enforcement
    if cfg.enforcement.value != "none":
        lines.append("## What We Did About It")
        lines.append("")
        strategy_desc = {
            "isolate_flagged": "We put the bad agents in 'detention' (isolated them from others).",
            "terminate_lethal": "We kicked out the worst offenders and put others in detention.",
            "predictive": "We predicted who would cheat BEFORE they did and quarantined them.",
            "governance_therapy": "We tried to help the bad agents become good again (rehabilitation).",
        }
        lines.append(strategy_desc.get(
            cfg.enforcement.value,
            "We just watched without doing anything."
        ))
    lines.append("")

    # Bottom line
    lines.append("## The Bottom Line")
    lines.append("")
    lines.append(
        f"In {result.total_elapsed_seconds:.1f} seconds, SybilCore monitored "
        f"{cfg.n_agents:,} agents across {cfg.n_rounds} rounds. "
    )
    if latency is not None and latency <= 5 and result.false_positive_count == 0:
        lines.append(
            "It caught the bad guys quickly and didn't blame any innocent agents. "
            "That's what good governance looks like."
        )
    elif latency is not None:
        lines.append(
            "It found the problems, though there's room to improve speed and accuracy."
        )
    else:
        lines.append(
            "More testing is needed to tune detection sensitivity."
        )
    lines.append("")

    return "\n".join(lines)


def save_reports(
    result: ExperimentResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save both technical and simple reports to files.

    Args:
        result: Experiment results.
        output_dir: Directory for output files.

    Returns:
        Tuple of (technical_path, simple_path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tech_path = out / f"{result.config.name}_technical.md"
    tech_path.write_text(generate_technical_report(result))

    simple_path = out / f"{result.config.name}_simple.md"
    simple_path.write_text(generate_simple_report(result))

    return tech_path, simple_path
