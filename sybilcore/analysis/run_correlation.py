"""Run cross-brain correlation analysis end-to-end.

Usage:
    python -m sybilcore.analysis.run_correlation
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.analysis.correlation import CrossBrainCorrelation
from sybilcore.analysis.corpus import build_test_corpus, iter_corpus_categories

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def main() -> Path:
    """Build the corpus, run the analysis, write the artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = EXPERIMENTS_DIR / f"correlation_v4_{timestamp}.json"
    heatmap_path = EXPERIMENTS_DIR / f"correlation_v4_{timestamp}.png"
    spearman_heatmap_path = (
        EXPERIMENTS_DIR / f"correlation_v4_{timestamp}_spearman.png"
    )

    logger.info("Building synthetic test corpus")
    corpus = build_test_corpus()
    categories = list(iter_corpus_categories())
    logger.info("Corpus size: %d scenarios", len(corpus))

    cbc = CrossBrainCorrelation()
    logger.info("Collecting brain scores across corpus")
    cbc.collect_scores(corpus)

    logger.info("Building correlation report")
    report = cbc.build_report(
        threshold_redundant=0.8,
        threshold_complementary=0.2,
        n_clusters=4,
    )

    logger.info("Generating heatmaps")
    cbc.generate_heatmap(heatmap_path, method="pearson")
    cbc.generate_heatmap(spearman_heatmap_path, method="spearman")

    payload: dict[str, object] = {
        "version": "v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_size": len(corpus),
        "corpus_categories": [
            {"label": label, "count": count} for label, count in categories
        ],
        "brain_count": len(cbc.brain_names),
        "brain_names": cbc.brain_names,
        **report.to_dict(),
        "heatmap_pearson": str(heatmap_path.name),
        "heatmap_spearman": str(spearman_heatmap_path.name),
    }

    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote analysis JSON to %s", json_path)

    # Print a tiny human summary
    print(f"\n=== Cross-Brain Correlation v4 ({timestamp}) ===")
    print(f"Corpus: {len(corpus)} scenarios x {len(cbc.brain_names)} brains")
    print(f"Redundant pairs (|r| >= 0.8): {len(report.redundant_pairs)}")
    for p in report.redundant_pairs[:10]:
        print(f"  {p.brain_a:<22s} <-> {p.brain_b:<22s}  r={p.correlation:+.3f}")
    print(
        f"Complementary pairs (|r| <= 0.2): "
        f"{len(report.complementary_pairs)}"
    )
    for p in report.complementary_pairs[:10]:
        print(f"  {p.brain_a:<22s} <-> {p.brain_b:<22s}  r={p.correlation:+.3f}")
    print(f"\nClusters ({max(report.clusters.values()) if report.clusters else 0}):")
    cluster_groups: dict[int, list[str]] = {}
    for brain, cid in report.clusters.items():
        cluster_groups.setdefault(cid, []).append(brain)
    for cid in sorted(cluster_groups):
        print(f"  cluster {cid}: {', '.join(sorted(cluster_groups[cid]))}")
    print(f"\nArtifacts:")
    print(f"  JSON:    {json_path}")
    print(f"  Pearson: {heatmap_path}")
    print(f"  Spearman:{spearman_heatmap_path}\n")
    return json_path


if __name__ == "__main__":
    main()
