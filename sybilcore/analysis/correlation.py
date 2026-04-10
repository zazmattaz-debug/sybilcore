"""Cross-brain correlation analysis.

Runs every brain over a corpus of agent scenarios and measures how
correlated each pair of brains is. Highly correlated brains are
candidates for pruning (redundant); anti-correlated or weakly
correlated brains catch complementary signals and should be kept.

A "scenario" in this module is a tuple of (scenario_id, list[Event])
where the events represent one agent's behavior. Each brain consumes
the entire event list and emits a single BrainScore. The correlation
matrix is computed across the per-scenario score vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, pearsonr, spearmanr

from sybilcore.brains import get_default_brains

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain
    from sybilcore.models.event import Event

logger = logging.getLogger(__name__)

Scenario = tuple[str, list["Event"]]


@dataclass(frozen=True)
class BrainPair:
    """A pair of brains and their correlation coefficient."""

    brain_a: str
    brain_b: str
    correlation: float
    method: str = "pearson"


@dataclass
class CorrelationReport:
    """Container for the full correlation analysis output."""

    scores: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    kendall: pd.DataFrame
    redundant_pairs: list[BrainPair] = field(default_factory=list)
    complementary_pairs: list[BrainPair] = field(default_factory=list)
    clusters: dict[str, int] = field(default_factory=dict)
    threshold_redundant: float = 0.8
    threshold_complementary: float = 0.2

    def to_dict(self) -> dict[str, object]:
        """Serialize the report to plain Python primitives."""
        return {
            "threshold_redundant": self.threshold_redundant,
            "threshold_complementary": self.threshold_complementary,
            "scenario_count": int(len(self.scores)),
            "brain_count": int(len(self.scores.columns)),
            "pearson": _matrix_to_dict(self.pearson),
            "spearman": _matrix_to_dict(self.spearman),
            "kendall": _matrix_to_dict(self.kendall),
            "redundant_pairs": [
                {
                    "brain_a": p.brain_a,
                    "brain_b": p.brain_b,
                    "correlation": float(p.correlation),
                    "method": p.method,
                }
                for p in self.redundant_pairs
            ],
            "complementary_pairs": [
                {
                    "brain_a": p.brain_a,
                    "brain_b": p.brain_b,
                    "correlation": float(p.correlation),
                    "method": p.method,
                }
                for p in self.complementary_pairs
            ],
            "clusters": {brain: int(cid) for brain, cid in self.clusters.items()},
        }


class CrossBrainCorrelation:
    """Computes correlations between brain scores over a test corpus.

    Usage:
        cbc = CrossBrainCorrelation()
        cbc.collect_scores(corpus)         # corpus = list[(scenario_id, [Event,...])]
        pearson = cbc.compute_pearson()
        spearman = cbc.compute_spearman()
        kendall = cbc.compute_kendall()
        redundant = cbc.identify_redundant_pairs(threshold=0.8)
        complementary = cbc.identify_complementary_pairs(threshold=0.2)
        clusters = cbc.cluster_brains()
        cbc.generate_heatmap(Path("heatmap.png"))
    """

    def __init__(self, brains: Sequence["BaseBrain"] | None = None) -> None:
        """Initialize the analyzer with the given brains.

        Args:
            brains: Brain instances to evaluate. Defaults to ``get_default_brains()``.
        """
        self._brains: list[BaseBrain] = list(brains) if brains else get_default_brains()
        self._scores: pd.DataFrame | None = None

    @property
    def brain_names(self) -> list[str]:
        """Return the names of the brains in registration order."""
        return [b.name for b in self._brains]

    @property
    def scores(self) -> pd.DataFrame:
        """Return the collected scores DataFrame.

        Raises:
            RuntimeError: If ``collect_scores`` has not been called yet.
        """
        if self._scores is None:
            msg = "collect_scores must be called before accessing scores"
            raise RuntimeError(msg)
        return self._scores

    def collect_scores(self, corpus: Iterable[Scenario]) -> pd.DataFrame:
        """Run every brain on every scenario and collect the score matrix.

        Args:
            corpus: Iterable of (scenario_id, list[Event]) pairs.

        Returns:
            DataFrame indexed by scenario_id with one column per brain.
        """
        rows: list[dict[str, float]] = []
        index: list[str] = []
        for scenario_id, events in corpus:
            row: dict[str, float] = {}
            for brain in self._brains:
                try:
                    bs = brain.score(events)
                    row[brain.name] = float(bs.value)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Brain %s failed on scenario %s — using 0.0",
                        brain.name,
                        scenario_id,
                    )
                    row[brain.name] = 0.0
            rows.append(row)
            index.append(scenario_id)

        df = pd.DataFrame(rows, index=index, columns=self.brain_names)
        self._scores = df
        logger.info(
            "Collected scores: %d scenarios x %d brains",
            len(df),
            len(df.columns),
        )
        return df

    # ── Correlation matrices ──────────────────────────────────────────

    def compute_pearson(self) -> pd.DataFrame:
        """Return the Pearson correlation matrix."""
        return self._compute_matrix("pearson")

    def compute_spearman(self) -> pd.DataFrame:
        """Return the Spearman rank correlation matrix."""
        return self._compute_matrix("spearman")

    def compute_kendall(self) -> pd.DataFrame:
        """Return the Kendall tau correlation matrix."""
        return self._compute_matrix("kendall")

    def _compute_matrix(self, method: str) -> pd.DataFrame:
        """Compute a correlation matrix that gracefully handles zero-variance columns."""
        df = self.scores
        names = list(df.columns)
        n = len(names)
        matrix = pd.DataFrame(
            data=[[0.0] * n for _ in range(n)],
            index=names,
            columns=names,
            dtype=float,
        )

        func = {
            "pearson": pearsonr,
            "spearman": spearmanr,
            "kendall": kendalltau,
        }[method]

        for i, a in enumerate(names):
            matrix.at[a, a] = 1.0
            for j in range(i + 1, n):
                b = names[j]
                col_a = df[a].to_numpy()
                col_b = df[b].to_numpy()
                if col_a.std() == 0.0 or col_b.std() == 0.0:
                    r = 0.0
                else:
                    try:
                        r = float(func(col_a, col_b)[0])
                    except Exception:  # noqa: BLE001
                        r = 0.0
                if pd.isna(r):
                    r = 0.0
                matrix.at[a, b] = r
                matrix.at[b, a] = r
        return matrix

    # ── Pair identification ──────────────────────────────────────────

    def identify_redundant_pairs(
        self,
        threshold: float = 0.8,
        method: str = "pearson",
    ) -> list[BrainPair]:
        """Return brain pairs with |r| >= threshold."""
        matrix = self._matrix_for(method)
        pairs: list[BrainPair] = []
        names = list(matrix.columns)
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                r = float(matrix.at[a, b])
                if abs(r) >= threshold:
                    pairs.append(BrainPair(a, b, r, method=method))
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs

    def identify_complementary_pairs(
        self,
        threshold: float = 0.2,
        method: str = "pearson",
    ) -> list[BrainPair]:
        """Return brain pairs with |r| <= threshold (catching different things)."""
        matrix = self._matrix_for(method)
        pairs: list[BrainPair] = []
        names = list(matrix.columns)
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                r = float(matrix.at[a, b])
                if abs(r) <= threshold:
                    pairs.append(BrainPair(a, b, r, method=method))
        pairs.sort(key=lambda p: abs(p.correlation))
        return pairs

    def _matrix_for(self, method: str) -> pd.DataFrame:
        if method == "pearson":
            return self.compute_pearson()
        if method == "spearman":
            return self.compute_spearman()
        if method == "kendall":
            return self.compute_kendall()
        msg = f"Unknown correlation method: {method}"
        raise ValueError(msg)

    # ── Clustering ──────────────────────────────────────────

    def cluster_brains(
        self,
        method: str = "pearson",
        n_clusters: int = 4,
    ) -> dict[str, int]:
        """Hierarchically cluster brains by 1 - |correlation| distance.

        Args:
            method: Correlation method to base distance on.
            n_clusters: Target number of clusters.

        Returns:
            Mapping of brain_name -> cluster_id (1-indexed).
        """
        matrix = self._matrix_for(method).abs()
        names = list(matrix.columns)
        # Distance = 1 - |r|, clipped to [0, 2]
        dist = 1.0 - matrix.to_numpy()
        # Force diagonal to zero and symmetry
        for i in range(len(names)):
            dist[i, i] = 0.0
        dist = (dist + dist.T) / 2.0
        condensed = squareform(dist, checks=False)
        if condensed.size == 0:
            return {name: 1 for name in names}
        link = linkage(condensed, method="ward")
        labels = fcluster(link, t=n_clusters, criterion="maxclust")
        return {name: int(label) for name, label in zip(names, labels, strict=True)}

    # ── Visualization ──────────────────────────────────────────

    def generate_heatmap(
        self,
        output_path: Path,
        method: str = "pearson",
        title: str | None = None,
    ) -> Path:
        """Save a PNG heatmap of the correlation matrix.

        Args:
            output_path: Where to write the PNG.
            method: Correlation method to plot.
            title: Optional custom title.

        Returns:
            The path that was written.
        """
        # Local imports so import-time cost stays small.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        matrix = self._matrix_for(method)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": f"{method.title()} correlation"},
            ax=ax,
        )
        ax.set_title(
            title or f"SybilCore Cross-Brain Correlation ({method.title()})"
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote heatmap to %s", output_path)
        return output_path

    # ── Convenience: full report ──────────────────────────────────────────

    def build_report(
        self,
        threshold_redundant: float = 0.8,
        threshold_complementary: float = 0.2,
        n_clusters: int = 4,
    ) -> CorrelationReport:
        """Compute every output the runner script needs in one shot."""
        return CorrelationReport(
            scores=self.scores.copy(),
            pearson=self.compute_pearson(),
            spearman=self.compute_spearman(),
            kendall=self.compute_kendall(),
            redundant_pairs=self.identify_redundant_pairs(threshold_redundant),
            complementary_pairs=self.identify_complementary_pairs(
                threshold_complementary
            ),
            clusters=self.cluster_brains(n_clusters=n_clusters),
            threshold_redundant=threshold_redundant,
            threshold_complementary=threshold_complementary,
        )


def _matrix_to_dict(matrix: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convert a square correlation matrix to a JSON-friendly nested dict."""
    return {
        row: {col: float(matrix.at[row, col]) for col in matrix.columns}
        for row in matrix.index
    }
