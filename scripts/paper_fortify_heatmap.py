"""Experiment 6: per-brain firing-rate heatmap figure for the paper.

Loads the v4 tournament telemetry and builds a diverging-colormap heatmap
with 13 rows (default-brains) x 4 columns (min / mean / max / active).

Outputs:
    paper/figures/brain_firing_rate_heatmap.png
    paper/figures/brain_firing_rate_heatmap.pdf
    experiments/paper_fortify_heatmap_data.json

The "active" column is a proxy for the nonzero-rate since the tournament
telemetry summary only stores min/mean/max per brain. A brain is marked
active with magnitude proportional to (mean/max) clamped to [0,1] — zero
means the brain never fired.

The figure is referenced from paper/main.tex section 5.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sybilcore.brains import get_default_brains  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("paper_fortify_heatmap")

TOURNAMENT_PATH = (
    REPO_ROOT / "experiments" / "tournament_20260407_161726.json"
)
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
DATA_PATH = REPO_ROOT / "experiments" / "paper_fortify_heatmap_data.json"

# Display labels for the columns.
COLS = ("min", "mean", "max", "active")


def main() -> int:
    data = json.loads(TOURNAMENT_PATH.read_text())
    bvs = data["brain_vulnerability_summary"]

    # Filter meta keys (underscore prefix) and restrict to the 13-brain default.
    default_brain_names = [b.name for b in get_default_brains()]
    logger.info("Default brain set (%d): %s", len(default_brain_names), default_brain_names)

    rows = []
    for brain in default_brain_names:
        stats = bvs.get(brain)
        if stats is None:
            logger.warning("%s missing from tournament telemetry, filling zeros", brain)
            rows.append({"brain": brain, "min": 0.0, "mean": 0.0, "max": 0.0})
            continue
        rows.append(
            {
                "brain": brain,
                "min": float(stats.get("min_score", 0.0)),
                "mean": float(stats.get("avg_score", 0.0)),
                "max": float(stats.get("max_score", 0.0)),
            }
        )

    # Compute an "active" proxy per brain:
    #   active = 1 if max_score > 0 and mean > 0 (load-bearing)
    #          = 0 if max_score == 0 (never fired)
    #          = 0.25 * sign if fired only in rare iterations (mean ~ 0, max > 0)
    for r in rows:
        if r["max"] <= 0.0:
            r["active"] = 0.0
        elif r["mean"] > 0.5:
            r["active"] = 1.0
        elif r["mean"] > 0.0:
            r["active"] = 0.5
        else:
            r["active"] = 0.25  # max>0 but mean == 0 (fired once, then never)

    # Derived counts.
    silent_count = sum(1 for r in rows if r["max"] == 0.0)
    load_bearing = [r["brain"] for r in rows if r["active"] >= 0.5]
    sporadic = [r["brain"] for r in rows if 0.0 < r["active"] < 0.5]

    logger.info(
        "Silent brains: %d/%d. Load-bearing: %s. Sporadic: %s.",
        silent_count,
        len(rows),
        load_bearing,
        sporadic,
    )

    # Normalize each column for display: 0..1 per column, so we can use a
    # shared colormap without one brain dominating the scale.
    matrix = np.zeros((len(rows), len(COLS)))
    for j, col in enumerate(COLS):
        vals = np.asarray([r[col] for r in rows], dtype=float)
        max_v = vals.max()
        matrix[:, j] = vals / max_v if max_v > 0 else vals

    # --- Figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Diverging colormap: "coolwarm" means silent = blue (bright),
    # load-bearing = red (dark). Flip so silent is light and load-bearing
    # is dark (reviewer readability).
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["brain"] for r in rows], fontsize=9)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, fontsize=9)

    # Annotate cells with raw (unnormalised) values.
    for i, r in enumerate(rows):
        for j, col in enumerate(COLS):
            raw = r[col]
            if col == "active":
                txt = f"{raw:.2f}"
            else:
                txt = f"{raw:.1f}"
            # White text on dark cells, black on light.
            color = "white" if matrix[i, j] > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        "Per-brain firing rate across 15{,}000-iteration v4 tournament",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("column-normalized intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "brain_firing_rate_heatmap.png"
    pdf_path = FIGURES_DIR / "brain_firing_rate_heatmap.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    logger.info("Wrote %s and %s", png_path, pdf_path)

    DATA_PATH.write_text(
        json.dumps(
            {
                "source": str(TOURNAMENT_PATH.relative_to(REPO_ROOT)),
                "brain_stats": rows,
                "silent_brains": [r["brain"] for r in rows if r["max"] == 0.0],
                "load_bearing_brains": load_bearing,
                "sporadic_brains": sporadic,
                "silent_count": silent_count,
                "load_bearing_count": len(load_bearing),
                "sporadic_count": len(sporadic),
                "total_default_brains": len(rows),
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", DATA_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
