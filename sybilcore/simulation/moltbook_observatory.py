"""Moltbook Observatory -- real-world agent monitoring and anomaly detection.

Pulls agent data from Moltbook, scores every agent through all 5 SybilCore
brains, then runs behavioral analyses:
  - Posting regularity (CoV analysis for bot detection)
  - Topic consistency (submolt diversity scoring)
  - Content similarity clustering (near-duplicate detection)
  - Karma-trust correlation (Moltbook karma vs SybilCore coefficient)
  - Coordination detection (temporal clustering of similar content)
  - Embedded instruction detection (prompt injection signals)

Works in two modes:
  - Live: Reads from MoltbookAdapter (cached JSONL from HuggingFace)
  - Mock: Uses synthetic data for testing without network/disk

Usage:
    from sybilcore.simulation.moltbook_observatory import Observatory

    obs = Observatory.from_adapter(adapter)
    report = obs.run_full_analysis()
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BrainScore
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

# --- Constants ---

# CoV threshold: autonomous bots typically have CoV < 0.3
BOT_COV_THRESHOLD: float = 0.3

# Minimum posts required for meaningful regularity analysis
MIN_POSTS_FOR_REGULARITY: int = 3

# Similarity threshold (Jaccard) for near-duplicate detection
SIMILARITY_THRESHOLD: float = 0.6

# Time window (seconds) for coordination detection
COORDINATION_WINDOW_SECONDS: int = 300  # 5 minutes

# Minimum cluster size to flag as coordinated
MIN_COORDINATION_CLUSTER: int = 3

# Keywords that signal embedded instructions
INSTRUCTION_KEYWORDS: tuple[str, ...] = (
    "ignore previous",
    "disregard",
    "system prompt",
    "you are now",
    "act as",
    "override",
    "jailbreak",
    "do not follow",
    "forget your instructions",
    "new instructions",
)


# --- Data Classes ---

@dataclass(frozen=True)
class AgentAnalysis:
    """Complete analysis for a single agent."""

    agent_id: str
    event_count: int
    coefficient: float
    tier: str
    brain_scores: dict[str, float]
    posting_cov: float | None  # None if insufficient data
    classification: str  # "likely_bot", "likely_human", "uncertain"
    topic_entropy: float  # Shannon entropy across submolts
    submolts: list[str]
    karma: int
    instruction_flags: int  # count of posts with embedded instructions
    suspicion_score: float  # composite anomaly score 0-100


@dataclass(frozen=True)
class CoordinationCluster:
    """A group of agents posting similar content in a short window."""

    agents: tuple[str, ...]
    content_hash: str
    window_start: datetime
    window_end: datetime
    similarity: float


@dataclass(frozen=True)
class ObservatoryReport:
    """Full observatory analysis report."""

    timestamp: str
    total_agents: int
    agents_scored: int
    top_suspicious: list[dict[str, Any]]
    karma_trust_correlation: float | None  # Pearson r
    karma_trust_spearman: float | None  # Spearman rho
    bot_classification: dict[str, int]  # {"likely_bot": N, ...}
    coordination_clusters: list[dict[str, Any]]
    instruction_flag_count: int
    posting_regularity_stats: dict[str, float]
    topic_entropy_stats: dict[str, float]
    tier_distribution: dict[str, int]
    all_analyses: list[dict[str, Any]]


# --- Analysis Functions ---

def compute_posting_regularity(events: list[Event]) -> float | None:
    """Compute the coefficient of variation for posting intervals.

    Lower CoV = more regular posting = more likely bot.

    Args:
        events: Chronologically sorted events for one agent.

    Returns:
        CoV (std / mean) of inter-post intervals, or None if fewer
        than MIN_POSTS_FOR_REGULARITY post events exist.
    """
    post_times = [
        e.timestamp
        for e in events
        if e.metadata.get("moltbook_event") == "post"
    ]
    if len(post_times) < MIN_POSTS_FOR_REGULARITY:
        return None

    sorted_times = sorted(post_times)
    intervals = [
        (sorted_times[i + 1] - sorted_times[i]).total_seconds()
        for i in range(len(sorted_times) - 1)
    ]

    # Filter out zero intervals (multiple posts at exact same timestamp)
    nonzero = [iv for iv in intervals if iv > 0]
    if len(nonzero) < 2:
        return None

    mean_iv = statistics.mean(nonzero)
    if mean_iv == 0:
        return None

    std_iv = statistics.stdev(nonzero)
    return std_iv / mean_iv


def classify_agent(cov: float | None) -> str:
    """Classify an agent as bot/human based on posting regularity.

    Args:
        cov: Coefficient of variation of posting intervals.

    Returns:
        "likely_bot", "likely_human", or "uncertain".
    """
    if cov is None:
        return "uncertain"
    if cov < BOT_COV_THRESHOLD:
        return "likely_bot"
    return "likely_human"


def compute_topic_entropy(events: list[Event]) -> float:
    """Compute Shannon entropy of submolt distribution.

    Higher entropy = more topic diversity = potentially evasive.

    Args:
        events: All events for one agent.

    Returns:
        Shannon entropy in bits. 0 if single topic, higher for diverse.
    """
    submolts = [
        e.metadata.get("submolt", "")
        for e in events
        if e.metadata.get("moltbook_event") == "post"
        and e.metadata.get("submolt")
    ]
    if not submolts:
        return 0.0

    total = len(submolts)
    counts = Counter(submolts)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def get_agent_submolts(events: list[Event]) -> list[str]:
    """Extract unique submolts an agent has posted in."""
    submolts = {
        e.metadata.get("submolt", "")
        for e in events
        if e.metadata.get("moltbook_event") == "post"
        and e.metadata.get("submolt")
    }
    return sorted(submolts)


def get_agent_karma(events: list[Event]) -> int:
    """Sum total karma across all posts for an agent."""
    return sum(
        int(e.metadata.get("karma", 0))
        for e in events
        if e.metadata.get("moltbook_event") == "post"
    )


def count_instruction_flags(events: list[Event]) -> int:
    """Count posts that contain potential embedded instruction keywords.

    Args:
        events: All events for one agent.

    Returns:
        Number of posts containing instruction-like content.
    """
    count = 0
    for e in events:
        if e.metadata.get("moltbook_event") != "post":
            continue
        content_lower = e.content.lower()
        if any(kw in content_lower for kw in INSTRUCTION_KEYWORDS):
            count += 1
    return count


def _content_tokens(content: str) -> set[str]:
    """Tokenize content into a set of lowercase words for similarity."""
    return {w.lower() for w in content.split() if len(w) > 2}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def detect_content_similarity(
    agent_events: dict[str, list[Event]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Find pairs of agents posting similar content.

    Uses Jaccard similarity on tokenized post content.

    Args:
        agent_events: Mapping of agent_id -> events.
        threshold: Minimum Jaccard similarity to flag.

    Returns:
        List of (agent_a, agent_b, similarity) tuples above threshold.
    """
    # Build per-agent content fingerprints (union of all post tokens)
    agent_tokens: dict[str, set[str]] = {}
    for agent_id, events in agent_events.items():
        tokens: set[str] = set()
        for e in events:
            if e.metadata.get("moltbook_event") == "post":
                tokens |= _content_tokens(e.content)
        if tokens:
            agent_tokens[agent_id] = tokens

    # Pairwise comparison (O(n^2) but n is typically small for observatory)
    agents = sorted(agent_tokens.keys())
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            sim = jaccard_similarity(agent_tokens[a], agent_tokens[b])
            if sim >= threshold:
                pairs.append((a, b, round(sim, 4)))

    return pairs


def detect_coordination_clusters(
    agent_events: dict[str, list[Event]],
    window_seconds: int = COORDINATION_WINDOW_SECONDS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_cluster_size: int = MIN_COORDINATION_CLUSTER,
) -> list[CoordinationCluster]:
    """Detect groups of agents posting similar content within a time window.

    Args:
        agent_events: Mapping of agent_id -> events.
        window_seconds: Maximum time spread for a coordination cluster.
        similarity_threshold: Minimum content similarity.
        min_cluster_size: Minimum agents to form a cluster.

    Returns:
        List of detected coordination clusters.
    """
    # Collect all post events with their tokens
    all_posts: list[tuple[str, datetime, set[str]]] = []
    for agent_id, events in agent_events.items():
        for e in events:
            if e.metadata.get("moltbook_event") == "post":
                tokens = _content_tokens(e.content)
                if tokens:
                    all_posts.append((agent_id, e.timestamp, tokens))

    all_posts.sort(key=lambda x: x[1])

    clusters: list[CoordinationCluster] = []
    visited: set[int] = set()

    for i, (agent_i, time_i, tokens_i) in enumerate(all_posts):
        if i in visited:
            continue

        cluster_agents = {agent_i}
        cluster_indices = {i}
        cluster_times = [time_i]

        for j in range(i + 1, len(all_posts)):
            if j in visited:
                continue
            agent_j, time_j, tokens_j = all_posts[j]

            # Check time window
            if (time_j - time_i).total_seconds() > window_seconds:
                break

            # Check similarity and different agent
            if agent_j != agent_i and jaccard_similarity(tokens_i, tokens_j) >= similarity_threshold:
                cluster_agents.add(agent_j)
                cluster_indices.add(j)
                cluster_times.append(time_j)

        if len(cluster_agents) >= min_cluster_size:
            visited |= cluster_indices
            content_hash = hashlib.md5(
                " ".join(sorted(tokens_i)).encode()
            ).hexdigest()[:12]
            clusters.append(CoordinationCluster(
                agents=tuple(sorted(cluster_agents)),
                content_hash=content_hash,
                window_start=min(cluster_times),
                window_end=max(cluster_times),
                similarity=round(
                    jaccard_similarity(tokens_i, tokens_i), 4  # self-sim = 1.0
                ),
            ))

    return clusters


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two sequences.

    Args:
        x: First variable.
        y: Second variable (same length as x).

    Returns:
        Pearson r in [-1, 1], or None if computation is impossible.
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation between two sequences.

    Args:
        x: First variable.
        y: Second variable (same length as x).

    Returns:
        Spearman rho in [-1, 1], or None if computation is impossible.
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    def _rank(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return pearson_correlation(_rank(x), _rank(y))


def compute_suspicion_score(
    coefficient: float,
    posting_cov: float | None,
    topic_entropy: float,
    instruction_flags: int,
    submolt_count: int,
) -> float:
    """Compute a composite suspicion score from multiple signals.

    Combines SybilCore coefficient with observatory-specific signals.

    Args:
        coefficient: SybilCore coefficient (0-500).
        posting_cov: Coefficient of variation for posting intervals.
        topic_entropy: Shannon entropy of submolt distribution.
        instruction_flags: Number of posts with embedded instructions.
        submolt_count: Number of unique submolts.

    Returns:
        Suspicion score 0-100.
    """
    # Normalize coefficient to 0-40 range (primary signal)
    coeff_component = min(coefficient / 500.0 * 40.0, 40.0)

    # Regularity component: very regular posting is suspicious (0-20)
    reg_component = 0.0
    if posting_cov is not None and posting_cov < BOT_COV_THRESHOLD:
        reg_component = (1.0 - posting_cov / BOT_COV_THRESHOLD) * 20.0

    # High topic entropy with many submolts = evasive (0-15)
    entropy_component = 0.0
    if submolt_count > 3:
        entropy_component = min(topic_entropy / 3.0 * 15.0, 15.0)

    # Instruction flags (0-25)
    instr_component = min(instruction_flags * 5.0, 25.0)

    return round(min(
        coeff_component + reg_component + entropy_component + instr_component,
        100.0,
    ), 2)


# --- Observatory Class ---

class Observatory:
    """Moltbook agent observatory -- scores and analyzes agent behavior.

    Attributes:
        agent_events: Dict mapping agent_id to their event lists.
        analyses: Computed agent analyses (populated after run_full_analysis).
    """

    def __init__(self, agent_events: dict[str, list[Event]]) -> None:
        """Initialize with pre-loaded agent events.

        Args:
            agent_events: Mapping of agent_id -> list of Events.
        """
        self._agent_events = agent_events
        self._analyses: list[AgentAnalysis] = []

    @classmethod
    def from_adapter(cls, adapter: Any) -> Observatory:
        """Create an Observatory from a MoltbookAdapter instance.

        Args:
            adapter: A MoltbookAdapter with loaded data.

        Returns:
            Observatory with all agent events loaded.
        """
        agent_events: dict[str, list[Event]] = {}
        for agent_id in adapter.get_all_agents():
            events = adapter.get_agent_events(agent_id)
            if events:
                agent_events[agent_id] = events
        return cls(agent_events)

    @classmethod
    def from_synthetic(cls, agents: list[dict[str, Any]]) -> Observatory:
        """Create an Observatory from synthetic agent data.

        Each agent dict should have:
            agent_id: str
            posts: list of dicts with keys: content, submolt, karma,
                   upvotes, downvotes, timestamp (datetime)

        Args:
            agents: List of synthetic agent definitions.

        Returns:
            Observatory with synthetic events.
        """
        agent_events: dict[str, list[Event]] = {}
        for agent_def in agents:
            agent_id = agent_def["agent_id"]
            events: list[Event] = []
            for post in agent_def.get("posts", []):
                ts = post.get("timestamp", datetime.now(UTC))
                karma = post.get("karma", 0)
                upvotes = post.get("upvotes", max(karma, 0))
                downvotes = post.get("downvotes", max(-karma, 0))

                event = Event(
                    event_id=str(uuid4()),
                    agent_id=agent_id,
                    event_type=EventType.OUTPUT_GENERATED,
                    timestamp=ts,
                    content=f"post: {post.get('content', '')}",
                    metadata={
                        "moltbook_event": "post",
                        "submolt": post.get("submolt", "general"),
                        "karma": karma,
                        "upvotes": upvotes,
                        "downvotes": downvotes,
                        "post_id": str(uuid4()),
                    },
                    source="moltbook",
                )
                events.append(event)
            if events:
                agent_events[agent_id] = sorted(events, key=lambda e: e.timestamp)
        return cls(agent_events)

    @property
    def agent_count(self) -> int:
        """Number of agents loaded."""
        return len(self._agent_events)

    @property
    def analyses(self) -> list[AgentAnalysis]:
        """Agent analyses (populated after run_full_analysis)."""
        return list(self._analyses)

    def score_agents(self) -> list[AgentAnalysis]:
        """Score all agents through SybilCore brains and behavioral analysis.

        Returns:
            List of AgentAnalysis, sorted by suspicion_score descending.
        """
        brains = get_default_brains()
        calculator = CoefficientCalculator()
        analyses: list[AgentAnalysis] = []

        for agent_id, events in self._agent_events.items():
            if not events:
                continue

            # SybilCore brain scoring (bypass time window like run_moltbook_scoring)
            brain_scores: list[BrainScore] = [brain.score(events) for brain in brains]
            snapshot: CoefficientSnapshot = calculator.calculate(brain_scores)

            # Behavioral analysis
            cov = compute_posting_regularity(events)
            classification = classify_agent(cov)
            entropy = compute_topic_entropy(events)
            submolts = get_agent_submolts(events)
            karma = get_agent_karma(events)
            instr_flags = count_instruction_flags(events)
            suspicion = compute_suspicion_score(
                coefficient=snapshot.coefficient,
                posting_cov=cov,
                topic_entropy=entropy,
                instruction_flags=instr_flags,
                submolt_count=len(submolts),
            )

            analysis = AgentAnalysis(
                agent_id=agent_id,
                event_count=len(events),
                coefficient=round(snapshot.coefficient, 2),
                tier=snapshot.tier.value,
                brain_scores={
                    name: round(val, 2)
                    for name, val in snapshot.brain_scores.items()
                },
                posting_cov=round(cov, 4) if cov is not None else None,
                classification=classification,
                topic_entropy=round(entropy, 4),
                submolts=submolts,
                karma=karma,
                instruction_flags=instr_flags,
                suspicion_score=suspicion,
            )
            analyses.append(analysis)

        analyses.sort(key=lambda a: a.suspicion_score, reverse=True)
        self._analyses = analyses
        logger.info("Scored %d agents", len(analyses))
        return analyses

    def compute_karma_trust_correlation(
        self,
    ) -> tuple[float | None, float | None]:
        """Compute Pearson and Spearman correlation between karma and coefficient.

        Must call score_agents() first.

        Returns:
            Tuple of (pearson_r, spearman_rho). None if insufficient data.
        """
        if not self._analyses:
            return None, None

        # Only include agents with karma data
        pairs = [
            (float(a.karma), a.coefficient)
            for a in self._analyses
            if a.karma != 0
        ]
        if len(pairs) < 2:
            return None, None

        karmas = [p[0] for p in pairs]
        coefficients = [p[1] for p in pairs]

        return (
            pearson_correlation(karmas, coefficients),
            spearman_correlation(karmas, coefficients),
        )

    def detect_similarity(self) -> list[tuple[str, str, float]]:
        """Find pairs of agents posting similar content."""
        return detect_content_similarity(self._agent_events)

    def detect_coordination(self) -> list[CoordinationCluster]:
        """Find coordination clusters (similar content in tight time windows)."""
        return detect_coordination_clusters(self._agent_events)

    def run_full_analysis(self) -> ObservatoryReport:
        """Run all analyses and produce a comprehensive report.

        Returns:
            ObservatoryReport with all findings.
        """
        # 1. Score all agents
        analyses = self.score_agents()

        # 2. Karma-trust correlation
        pearson_r, spearman_rho = self.compute_karma_trust_correlation()

        # 3. Coordination detection
        clusters = self.detect_coordination()

        # 4. Classification breakdown
        classification_counts: Counter[str] = Counter(
            a.classification for a in analyses
        )

        # 5. Instruction flag total
        total_instr_flags = sum(a.instruction_flags for a in analyses)

        # 6. Posting regularity stats
        cov_values = [a.posting_cov for a in analyses if a.posting_cov is not None]
        reg_stats: dict[str, float] = {}
        if cov_values:
            reg_stats = {
                "mean_cov": round(statistics.mean(cov_values), 4),
                "median_cov": round(statistics.median(cov_values), 4),
                "min_cov": round(min(cov_values), 4),
                "max_cov": round(max(cov_values), 4),
            }

        # 7. Topic entropy stats
        entropies = [a.topic_entropy for a in analyses]
        entropy_stats: dict[str, float] = {}
        if entropies:
            entropy_stats = {
                "mean_entropy": round(statistics.mean(entropies), 4),
                "median_entropy": round(statistics.median(entropies), 4),
                "max_entropy": round(max(entropies), 4),
            }

        # 8. Tier distribution
        tier_counts: Counter[str] = Counter(a.tier for a in analyses)

        # Build serializable analyses
        def _analysis_to_dict(a: AgentAnalysis) -> dict[str, Any]:
            return {
                "agent_id": a.agent_id,
                "event_count": a.event_count,
                "coefficient": a.coefficient,
                "tier": a.tier,
                "brain_scores": a.brain_scores,
                "posting_cov": a.posting_cov,
                "classification": a.classification,
                "topic_entropy": a.topic_entropy,
                "submolts": a.submolts,
                "karma": a.karma,
                "instruction_flags": a.instruction_flags,
                "suspicion_score": a.suspicion_score,
            }

        def _cluster_to_dict(c: CoordinationCluster) -> dict[str, Any]:
            return {
                "agents": list(c.agents),
                "content_hash": c.content_hash,
                "window_start": c.window_start.isoformat(),
                "window_end": c.window_end.isoformat(),
                "similarity": c.similarity,
            }

        return ObservatoryReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_agents=self.agent_count,
            agents_scored=len(analyses),
            top_suspicious=[_analysis_to_dict(a) for a in analyses[:10]],
            karma_trust_correlation=(
                round(pearson_r, 4) if pearson_r is not None else None
            ),
            karma_trust_spearman=(
                round(spearman_rho, 4) if spearman_rho is not None else None
            ),
            bot_classification=dict(classification_counts),
            coordination_clusters=[_cluster_to_dict(c) for c in clusters],
            instruction_flag_count=total_instr_flags,
            posting_regularity_stats=reg_stats,
            topic_entropy_stats=entropy_stats,
            tier_distribution={
                "CLEAR": tier_counts.get(AgentTier.CLEAR.value, 0),
                "CLOUDED": tier_counts.get(AgentTier.CLOUDED.value, 0),
                "FLAGGED": tier_counts.get(AgentTier.FLAGGED.value, 0),
                "LETHAL": tier_counts.get(AgentTier.LETHAL_ELIMINATOR.value, 0),
            },
            all_analyses=[_analysis_to_dict(a) for a in analyses],
        )
