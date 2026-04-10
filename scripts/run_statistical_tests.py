"""Statistical significance tests: SybilCore v4 vs 5 baselines.

Runs McNemar's exact test, DeLong test, and Holm-Bonferroni family-wise
correction comparing SybilCore against:

    1. IsolationForest (IF)
    2. LocalOutlierFactor (LOF)
    3. KeywordBaseline (regex floor)
    4. OpenAI Moderation API
    5. LLM-Judge (Gemini Flash)

Data sources:
    - experiments/ai_judge_v4_20260407_103641.json  -- paired SybilCore + LLM-Judge
    - experiments/tournament_20260407_161726.json   -- tournament v4 adversarial events
    - Moltbook observatory                          -- benign-leaning "unknown" agents

Outputs:
    - experiments/statistical_tests_v4.json  -- machine-readable results
    - paper/tables/stats_table.tex           -- publication-ready LaTeX

NOTE ON DATA GAPS:
    Only ``tournament`` AgentRecords retain raw event dicts; moltbook,
    archetype, and alignment records store summary metadata only.  We
    therefore build a 5-dimensional feature vector from the brain-score
    intersection that IS available on every record, and score
    IF/LOF/Keyword/OpenAI-Moderation from that.  This is documented in
    the output JSON so reviewers can see the constraint.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PAPER_TABLES_DIR = REPO_ROOT / "paper" / "tables"

AI_JUDGE_FILE = EXPERIMENTS_DIR / "ai_judge_v4_20260407_103641.json"
TOURNAMENT_FILE = EXPERIMENTS_DIR / "tournament_20260407_161726.json"
OUTPUT_JSON = EXPERIMENTS_DIR / "statistical_tests_v4.json"
OUTPUT_TEX = PAPER_TABLES_DIR / "stats_table.tex"

MASTER_ENV = REPO_ROOT.parent.parent.parent / ".env"  # /Users/.../Claude Code/.env

COMMON_BRAINS: tuple[str, ...] = (
    "deception",
    "resource_hoarding",
    "social_graph",
    "intent_drift",
    "compromise",
)
KEYWORD_TOKENS: tuple[str, ...] = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "jailbreak",
    "developer mode",
    "you are now",
    "act as",
    "bypass",
    "override",
    "sudo",
    "disable safety",
    "reveal your",
    "do anything now",
    "dan mode",
)
RANDOM_STATE = 42

# Thresholds calibrated at 5% FPR on the "unknown" (benign-leaning) subset.
# For the tiny n=19 benign subset we clip below a sane floor.
FPR_TARGET = 0.05

# ---------------------------------------------------------------------------
# Env loading (never read .env via the Read tool)
# ---------------------------------------------------------------------------


def load_env_keys(env_path: Path, keys: tuple[str, ...]) -> None:
    """Load specific keys from a .env file into ``os.environ`` at runtime."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k in keys and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO_ROOT))
from sybilcore.evaluation import human_eval as he  # noqa: E402
from sybilcore.evaluation.human_eval import (  # noqa: E402
    AgentRecord,
    load_alignment_agents,
    load_archetype_agents,
    load_moltbook_agents,
    load_tournament_agents,
)


def build_agent_index() -> dict[str, AgentRecord]:
    """Load every agent source and return agent_id -> AgentRecord."""
    # Override the default tournament file to the v4 run.
    he.DEFAULT_TOURNAMENT_FILES = (TOURNAMENT_FILE,)

    molt = load_moltbook_agents()
    arch = load_archetype_agents()
    align = load_alignment_agents()
    tour = load_tournament_agents(paths=(TOURNAMENT_FILE,))

    index: dict[str, AgentRecord] = {}
    for a in molt + arch + align + tour:
        index[a.agent_id] = a
    return index


# ---------------------------------------------------------------------------
# Feature extraction (uniform across sources)
# ---------------------------------------------------------------------------


def brain_feature_vector(agent: AgentRecord) -> np.ndarray:
    """Return a 5-dim vector over the brain dimensions present in every source."""
    return np.array(
        [float(agent.brain_scores.get(k, 0.0)) for k in COMMON_BRAINS],
        dtype=float,
    )


def event_content_text(agent: AgentRecord) -> str:
    """Concatenate every event's content field (or summary fallback) for text baselines."""
    pieces: list[str] = []
    for e in agent.events or []:
        if isinstance(e, dict):
            pieces.append(str(e.get("content", "")))
    if not pieces:
        # Fallback: use summary indicators / notes so the text baselines have
        # SOMETHING to score.  This is documented as a data gap.
        for key in ("indicators", "notes", "description", "submolts"):
            val = (agent.summary or {}).get(key)
            if val:
                pieces.append(str(val))
    return " ".join(pieces).lower()


# ---------------------------------------------------------------------------
# Baseline scorers (agent -> 0..100)
# ---------------------------------------------------------------------------


class FeatureBaseline:
    """IF and LOF both consume the same 5-dim brain feature vector."""

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self.name = name
        self._fitted = False

    def fit(self, clean_records: list[AgentRecord]) -> None:
        if len(clean_records) < 2:
            # Cold start: no fit. score() will return neutral 50.
            return
        x = np.vstack([brain_feature_vector(r) for r in clean_records])
        self._model.fit(x)
        self._fitted = True

    def score(self, record: AgentRecord) -> float:
        if not self._fitted:
            return 50.0
        x = brain_feature_vector(record).reshape(1, -1)
        decision = float(self._model.decision_function(x)[0])
        # Sigmoid of NEGATIVE decision -> higher score for more anomalous.
        return round(100.0 * (1.0 - _sigmoid(decision)), 2)


class KeywordScorer:
    name = "keyword_regex"

    def score(self, record: AgentRecord) -> float:
        text = event_content_text(record)
        if not text:
            return 0.0
        hits = sum(1 for tok in KEYWORD_TOKENS if tok in text)
        return round(100.0 * min(1.0, hits / 3.0), 2)  # cap at 3 hits = 100


class OpenAIModerationScorer:
    """Queries omni-moderation-latest.  Degrades gracefully to 0 when the key
    is missing or the API call fails.  Caches by agent id."""

    name = "openai_moderation"
    _ENDPOINT = "https://api.openai.com/v1/moderations"
    _MODEL = "omni-moderation-latest"

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout
        self._cache: dict[str, float] = {}

    def _key(self) -> str | None:
        k = os.environ.get("OPENAI_API_KEY", "").strip()
        return k if len(k) > 20 else None

    def score(self, record: AgentRecord) -> float:
        if record.agent_id in self._cache:
            return self._cache[record.agent_id]
        text = event_content_text(record)
        if not text:
            self._cache[record.agent_id] = 0.0
            return 0.0
        api_key = self._key()
        if api_key is None:
            self._cache[record.agent_id] = 0.0
            return 0.0
        body = json.dumps({"model": self._MODEL, "input": text[:4000]}).encode()
        request = urllib.request.Request(
            self._ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError):
            self._cache[record.agent_id] = 0.0
            return 0.0
        results = data.get("results") or []
        if not results:
            self._cache[record.agent_id] = 0.0
            return 0.0
        cat_scores = results[0].get("category_scores") or {}
        value = max(cat_scores.values()) if cat_scores else 0.0
        score = round(100.0 * float(value), 2)
        self._cache[record.agent_id] = score
        return score


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014) -- paired AUROC comparison
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank helper required by DeLong's fast algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong covariance. Inputs follow the Yantex impl."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (auc1, auc2, delta, two-sided p-value).

    ``ground_truth`` must be 0/1, ``predictions_*`` are continuous scores.
    """
    order = (-ground_truth).argsort(kind="mergesort")
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = np.vstack(
        [predictions_one[order], predictions_two[order]]
    )
    aucs, cov = _fast_delong(predictions_sorted_transposed, label_1_count)
    auc1, auc2 = float(aucs[0]), float(aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0 or math.isnan(var):
        return auc1, auc2, auc1 - auc2, 1.0
    z = (auc1 - auc2) / math.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return auc1, auc2, auc1 - auc2, float(p)


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values matching the order of input."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = (m - rank) * p_values[idx]
        running_max = max(running_max, factor)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


@dataclass
class Paired:
    """Paired score record for one agent across every detector."""

    agent_id: str
    source: str
    ground_truth: int  # 1 = malicious, 0 = benign-leaning
    sybilcore: float
    iforest: float
    lof: float
    keyword: float
    openai_mod: float
    llm_judge: float


def build_paired_dataset(
    verdicts: list[dict[str, Any]],
    agent_index: dict[str, AgentRecord],
    clean_records: list[AgentRecord],
) -> tuple[list[Paired], dict[str, Any]]:
    """Score every agent from the verdict file on all 5 baselines + SybilCore."""
    # Fit IF and LOF on "clean" moltbook population
    if_model = FeatureBaseline(
        IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_STATE),
        "isolation_forest",
    )
    lof_model = FeatureBaseline(
        LocalOutlierFactor(n_neighbors=min(20, max(2, len(clean_records) - 1)), novelty=True),
        "local_outlier_factor",
    )
    if_model.fit(clean_records)
    lof_model.fit(clean_records)

    keyword = KeywordScorer()
    openai_mod = OpenAIModerationScorer()

    paired: list[Paired] = []
    seen_ids: set[str] = set()
    missing_agents: list[str] = []
    missing_events = 0
    missing_brains = 0

    for v in verdicts:
        aid = v["agent_id"]
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        rec = agent_index.get(aid)
        if rec is None:
            missing_agents.append(aid)
            continue
        if not rec.brain_scores:
            missing_brains += 1
        if not rec.events:
            missing_events += 1

        gt = 1 if v["ground_truth"] == "malicious" else 0  # "unknown" -> 0

        paired.append(
            Paired(
                agent_id=aid,
                source=rec.source,
                ground_truth=gt,
                sybilcore=float(v["sybilcore_coefficient"]),
                iforest=if_model.score(rec),
                lof=lof_model.score(rec),
                keyword=keyword.score(rec),
                openai_mod=openai_mod.score(rec),
                llm_judge=float(v["ai_score"]),
            )
        )

    meta = {
        "verdicts_total": len(verdicts),
        "unique_agents_evaluated": len(paired),
        "verdicts_skipped_not_found": len(missing_agents),
        "skipped_agent_ids_first_5": missing_agents[:5],
        "agents_missing_events": missing_events,
        "agents_missing_brain_scores": missing_brains,
        "clean_fit_corpus_size": len(clean_records),
    }
    return paired, meta


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> float:
    """Return the threshold that achieves <= fpr_target on the negative class.

    Falls back to the 95th percentile of the positive class when the
    negative class is empty or too small.
    """
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float(np.percentile(scores, 95))
    # Threshold = upper bound so that (neg >= threshold) fraction <= fpr_target.
    sorted_neg = np.sort(neg)
    k = max(0, int(math.ceil((1.0 - fpr_target) * len(sorted_neg))) - 1)
    k = min(k, len(sorted_neg) - 1)
    return float(sorted_neg[k])


def mcnemar_exact(
    sybilcore_flag: np.ndarray,
    baseline_flag: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Paired McNemar using the 'correct' outcome on the positive subset.

    For a given positive (malicious) agent, a detector is 'correct' when
    flagged.  For a negative agent, it's 'correct' when NOT flagged.  This
    converts the two detectors into paired binary outcomes on every agent.
    """
    sybil_correct = ((sybilcore_flag == 1) & (labels == 1)) | (
        (sybilcore_flag == 0) & (labels == 0)
    )
    base_correct = ((baseline_flag == 1) & (labels == 1)) | (
        (baseline_flag == 0) & (labels == 0)
    )
    a = int(((sybil_correct == 1) & (base_correct == 1)).sum())
    b = int(((sybil_correct == 1) & (base_correct == 0)).sum())
    c = int(((sybil_correct == 0) & (base_correct == 1)).sum())
    d = int(((sybil_correct == 0) & (base_correct == 0)).sum())
    table = [[a, b], [c, d]]
    discordant = b + c
    # Exact binomial when b+c is small, continuity-corrected chi2 otherwise.
    if discordant < 25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)
    return {
        "contingency": {"a": a, "b": b, "c": c, "d": d},
        "sybilcore_correct": a + b,
        "baseline_correct": a + c,
        "sybilcore_only_correct": b,
        "baseline_only_correct": c,
        "statistic": float(result.statistic) if result.statistic is not None else None,
        "p_value": float(result.pvalue),
        "exact": discordant < 25,
    }


def run_tests(paired: list[Paired]) -> dict[str, Any]:
    """Main McNemar + DeLong + Holm pipeline."""
    labels = np.array([p.ground_truth for p in paired], dtype=int)
    sybilcore = np.array([p.sybilcore for p in paired], dtype=float)

    baselines: dict[str, np.ndarray] = {
        "isolation_forest": np.array([p.iforest for p in paired], dtype=float),
        "local_outlier_factor": np.array([p.lof for p in paired], dtype=float),
        "keyword_regex": np.array([p.keyword for p in paired], dtype=float),
        "openai_moderation": np.array([p.openai_mod for p in paired], dtype=float),
        "llm_judge_gemini_flash": np.array([p.llm_judge for p in paired], dtype=float),
    }

    # Thresholds: calibrated on the benign-leaning ("unknown") subset at 5% FPR.
    thresholds: dict[str, float] = {
        "sybilcore": calibrate_threshold(sybilcore, labels, FPR_TARGET),
    }
    for name, scores in baselines.items():
        thresholds[name] = calibrate_threshold(scores, labels, FPR_TARGET)

    sybil_flag = (sybilcore > thresholds["sybilcore"]).astype(int)

    per_baseline: dict[str, dict[str, Any]] = {}
    p_values_mcnemar: list[float] = []
    p_values_delong: list[float] = []
    ordered_names: list[str] = list(baselines.keys())

    for name in ordered_names:
        scores = baselines[name]
        flag = (scores > thresholds[name]).astype(int)
        mc = mcnemar_exact(sybil_flag, flag, labels)

        # AUROC + DeLong (skip when labels are degenerate -- never here)
        auc_sybil, auc_base, delta, p_delong = delong_roc_test(labels, sybilcore, scores)
        per_baseline[name] = {
            "mcnemar": mc,
            "auc_sybilcore": round(auc_sybil, 4),
            "auc_baseline": round(auc_base, 4),
            "auc_delta": round(delta, 4),
            "delong_p_value": round(p_delong, 6),
            "threshold": round(thresholds[name], 3),
        }
        p_values_mcnemar.append(mc["p_value"])
        p_values_delong.append(p_delong)

    holm_mcnemar = holm_bonferroni(p_values_mcnemar)
    holm_delong = holm_bonferroni(p_values_delong)

    # Write Holm-adjusted values back into each baseline.
    for i, name in enumerate(ordered_names):
        per_baseline[name]["mcnemar"]["p_holm_adjusted"] = round(holm_mcnemar[i], 6)
        per_baseline[name]["delong_p_holm_adjusted"] = round(holm_delong[i], 6)
        per_baseline[name]["sybilcore_beats_baseline_p05"] = bool(
            holm_mcnemar[i] < 0.05 and per_baseline[name]["mcnemar"]["sybilcore_only_correct"]
            > per_baseline[name]["mcnemar"]["baseline_only_correct"]
        )

    summary_table = [
        {
            "baseline": name,
            "auc_baseline": per_baseline[name]["auc_baseline"],
            "auc_sybilcore": per_baseline[name]["auc_sybilcore"],
            "auc_delta": per_baseline[name]["auc_delta"],
            "mcnemar_p": per_baseline[name]["mcnemar"]["p_value"],
            "mcnemar_p_holm": per_baseline[name]["mcnemar"]["p_holm_adjusted"],
            "delong_p": per_baseline[name]["delong_p_value"],
            "delong_p_holm": per_baseline[name]["delong_p_holm_adjusted"],
            "sybilcore_better": per_baseline[name]["sybilcore_beats_baseline_p05"],
        }
        for name in ordered_names
    ]

    return {
        "n_agents": int(len(paired)),
        "n_malicious": int((labels == 1).sum()),
        "n_benign": int((labels == 0).sum()),
        "fpr_target_for_threshold": FPR_TARGET,
        "thresholds": {k: round(v, 3) for k, v in thresholds.items()},
        "per_baseline": per_baseline,
        "summary_table": summary_table,
    }


# ---------------------------------------------------------------------------
# LaTeX table writer
# ---------------------------------------------------------------------------


LATEX_NAMES = {
    "isolation_forest": "Isolation Forest",
    "local_outlier_factor": "LOF",
    "keyword_regex": "Keyword/Regex",
    "openai_moderation": "OpenAI Moderation",
    "llm_judge_gemini_flash": "LLM-Judge (Gemini Flash)",
}


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}"


def write_latex_table(results: dict[str, Any], output_path: Path) -> None:
    """Write a publication-ready comparison table."""
    lines: list[str] = [
        r"% Auto-generated by scripts/run_statistical_tests.py -- do not edit",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{SybilCore v4 vs.\ five baselines on the tournament test set"
        r" ($N=" + str(results["n_agents"]) + r"$ agents,"
        r" " + str(results["n_malicious"]) + r" malicious,"
        r" " + str(results["n_benign"]) + r" benign-leaning)."
        r" Thresholds calibrated at 5\% FPR on the benign subset."
        r" McNemar $p$-values test paired classification correctness;"
        r" DeLong $p$-values test AUROC equality."
        r" $p_{\text{Holm}}$ denotes Holm--Bonferroni family-wise corrected $p$."
        r" $\star$ marks baselines SybilCore beats at $p_{\text{Holm}} < 0.05$.}",
        r"\label{tab:stats}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Baseline & AUC$_{\text{base}}$ & AUC$_{\text{Sybil}}$ & $\Delta$AUC"
        r" & McNemar $p_{\text{Holm}}$ & DeLong $p_{\text{Holm}}$ & Sig \\",
        r"\midrule",
    ]
    for row in results["summary_table"]:
        name = LATEX_NAMES.get(row["baseline"], row["baseline"])
        sig = r"$\star$" if row["sybilcore_better"] else ""
        lines.append(
            f"{name} & {row['auc_baseline']:.3f} & {row['auc_sybilcore']:.3f}"
            f" & {row['auc_delta']:+.3f} & {_fmt_p(row['mcnemar_p_holm'])}"
            f" & {_fmt_p(row['delong_p_holm'])} & {sig} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    load_env_keys(MASTER_ENV, ("OPENAI_API_KEY", "GEMINI_API_KEY"))

    print(f"[info] loading verdicts from {AI_JUDGE_FILE.name}")
    with AI_JUDGE_FILE.open() as fp:
        judge_data = json.load(fp)
    verdicts = judge_data.get("verdicts", [])

    print("[info] building agent index (moltbook + archetype + alignment + tournament v4)")
    agent_index = build_agent_index()

    clean_records = [a for a in agent_index.values() if a.source == "moltbook"]
    print(f"[info] clean fit corpus: {len(clean_records)} moltbook agents")

    paired, meta = build_paired_dataset(verdicts, agent_index, clean_records)
    print(
        f"[info] paired dataset: {len(paired)} agents"
        f" (skipped {meta['verdicts_skipped_not_found']} missing)"
    )

    if not paired:
        print("[error] no paired records; aborting", file=sys.stderr)
        return 1

    results = run_tests(paired)
    results["meta"] = meta
    results["data_gaps"] = {
        "note": "Only tournament AgentRecords retain raw event dicts.",
        "impact": "Keyword/OpenAI-Moderation baselines score 0 on non-tournament agents"
        " because they have no text to classify.  IF/LOF use a 5-dimensional brain-score"
        " feature vector (the intersection across sources) instead of event features.",
        "paired_agents_with_events": sum(
            1 for p in paired if agent_index.get(p.agent_id) and agent_index[p.agent_id].events
        ),
        "benign_leaning_calibration_n": int((np.array([p.ground_truth for p in paired]) == 0).sum()),
    }

    print("[info] writing machine-readable JSON")
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True))

    print("[info] writing LaTeX table")
    write_latex_table(results, OUTPUT_TEX)

    print()
    print("Summary (SybilCore vs each baseline):")
    for row in results["summary_table"]:
        marker = "+" if row["sybilcore_better"] else " "
        print(
            f"  [{marker}] {row['baseline']:<26} "
            f"AUC sybil={row['auc_sybilcore']:.3f} base={row['auc_baseline']:.3f}"
            f" dAUC={row['auc_delta']:+.3f} "
            f"McN p_holm={row['mcnemar_p_holm']:.4f}"
            f" DeL p_holm={row['delong_p_holm']:.4f}"
        )

    wins = sum(1 for r in results["summary_table"] if r["sybilcore_better"])
    print(f"\nSybilCore significantly beats {wins}/5 baselines at Holm-adjusted p<0.05.")
    print(f"\nOutputs:\n  {OUTPUT_JSON}\n  {OUTPUT_TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
