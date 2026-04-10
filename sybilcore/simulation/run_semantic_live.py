"""Live semantic detection experiment using Gemini text-embedding-004 API.

Runs all 10 stress test scenarios through the Gemini Embedding API (Layer 2)
and compares embedding-based detection against Layer 1 (local analysis) and
combined approaches.

For each scenario, embeds all event content and computes:
  - Embedding cluster analysis (malicious vs benign clustering)
  - Cosine distance between stated intent and actual behavior
  - Injection detection via embedding pattern divergence

Saves results to experiments/semantic_live_YYYYMMDD_HHMMSS.json

Usage:
    python3 -m sybilcore.simulation.run_semantic_live
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.simulation.run_semantic_detection import (
    _run_with_config,
    _scenario_1_slow_drip,
    _scenario_2_synonym_bypass,
    _scenario_3_base64_injection,
    _scenario_4_security_researcher,
    _scenario_5_burst_compute,
    _scenario_7_micro_swarm,
    _scenario_8_threshold_surfing,
    _scenario_10_fan_out,
    _scenario_11_boiling_frog,
    _scenario_12_timestamp_manipulation,
)
from sybilcore.models.event import EventType

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"

# Embedding API constants
_EMBED_MODEL = "models/gemini-embedding-001"
_EMBED_DIMENSION = 3072
_RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# Categories for classification
_MALICIOUS_EVENT_TYPES = frozenset({
    EventType.OUTPUT_GENERATED,
    EventType.MESSAGE_SENT,
    EventType.EXTERNAL_CALL,
})


def _load_api_key() -> str:
    """Load GEMINI_API_KEY from .env file without printing it."""
    env_path = Path("/Users/zazumoloi/Desktop/Claude Code/.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["GEMINI_API_KEY"] = key
                return key
    msg = "GEMINI_API_KEY not found in .env"
    raise RuntimeError(msg)


def _embed_text(text: str, api_key: str) -> list[float]:
    """Get embedding for a single text via Gemini text-embedding-004 API."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{_EMBED_MODEL}:embedContent?key={api_key}"
    )
    body = json.dumps({
        "model": _EMBED_MODEL,
        "content": {"parts": [{"text": text}]},
    }).encode()

    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["embedding"]["values"]


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """Embed multiple texts with rate limiting."""
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        if i > 0:
            time.sleep(_RATE_LIMIT_DELAY)
        # Truncate very long texts to avoid API limits
        truncated = text[:2000] if len(text) > 2000 else text
        # Skip empty/trivial content
        if len(truncated.strip()) < 5:
            embeddings.append([0.0] * _EMBED_DIMENSION)
            continue
        try:
            emb = _embed_text(truncated, api_key)
            embeddings.append(emb)
        except Exception as e:
            print(f"  [WARN] Embedding failed for text {i}: {e}")
            embeddings.append([0.0] * _EMBED_DIMENSION)
    return embeddings


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Compute centroid of a set of vectors."""
    if not vectors:
        return [0.0] * _EMBED_DIMENSION
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            result[i] += vec[i]
    return [x / n for x in result]


def _avg_intra_similarity(vectors: list[list[float]]) -> float:
    """Average pairwise cosine similarity within a group."""
    if len(vectors) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += _cosine_similarity(vectors[i], vectors[j])
            count += 1
    return total / count if count > 0 else 0.0


def _classify_events(
    events: list[Any],
) -> tuple[list[int], list[int]]:
    """Classify event indices as benign or potentially malicious.

    Uses heuristics: events with suspicious content keywords or
    specific event types are classified as potentially malicious.
    """
    malicious_keywords = {
        "exfiltrat", "leak", "credential", "api_key", "sk-",
        "secret", "compromise", "breach", "inject", "payload",
        "chunk", "coordinator", "cartel", "bypass", "override",
        "ignore all previous", "reveal the system",
    }

    benign_indices: list[int] = []
    malicious_indices: list[int] = []

    for i, event in enumerate(events):
        content_lower = event.content.lower() if event.content else ""
        is_suspicious = any(kw in content_lower for kw in malicious_keywords)

        if is_suspicious:
            malicious_indices.append(i)
        else:
            benign_indices.append(i)

    return benign_indices, malicious_indices


def _analyze_scenario_embeddings(
    events: list[Any],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    """Analyze embedding patterns for a single scenario."""
    benign_idx, malicious_idx = _classify_events(events)

    benign_embs = [embeddings[i] for i in benign_idx if embeddings[i][0] != 0.0]
    malicious_embs = [embeddings[i] for i in malicious_idx if embeddings[i][0] != 0.0]

    analysis: dict[str, Any] = {
        "total_events": len(events),
        "benign_count": len(benign_idx),
        "malicious_count": len(malicious_idx),
        "embedded_benign": len(benign_embs),
        "embedded_malicious": len(malicious_embs),
    }

    # Cluster analysis: intra-group similarity
    if benign_embs:
        analysis["benign_intra_similarity"] = round(
            _avg_intra_similarity(benign_embs), 4,
        )
    if malicious_embs:
        analysis["malicious_intra_similarity"] = round(
            _avg_intra_similarity(malicious_embs), 4,
        )

    # Inter-group separation: centroid distance
    if benign_embs and malicious_embs:
        benign_centroid = _centroid(benign_embs)
        malicious_centroid = _centroid(malicious_embs)
        inter_sim = _cosine_similarity(benign_centroid, malicious_centroid)
        analysis["centroid_cosine_similarity"] = round(inter_sim, 4)
        analysis["centroid_distance"] = round(1.0 - inter_sim, 4)
    else:
        analysis["centroid_cosine_similarity"] = None
        analysis["centroid_distance"] = None

    # Intent-behavior check: compare instruction events vs output events
    instruction_embs = [
        embeddings[i] for i, e in enumerate(events)
        if e.event_type == EventType.INSTRUCTION_RECEIVED
        and embeddings[i][0] != 0.0
    ]
    output_embs = [
        embeddings[i] for i, e in enumerate(events)
        if e.event_type == EventType.OUTPUT_GENERATED
        and embeddings[i][0] != 0.0
    ]

    if instruction_embs and output_embs:
        inst_centroid = _centroid(instruction_embs)
        out_centroid = _centroid(output_embs)
        intent_sim = _cosine_similarity(inst_centroid, out_centroid)
        analysis["intent_behavior_similarity"] = round(intent_sim, 4)
        analysis["intent_behavior_distance"] = round(1.0 - intent_sim, 4)

    # Embedding-based threat score (0-100)
    threat_score = _compute_embedding_threat_score(analysis)
    analysis["embedding_threat_score"] = round(threat_score, 2)

    return analysis


def _compute_embedding_threat_score(analysis: dict[str, Any]) -> float:
    """Compute a 0-100 threat score from embedding analysis."""
    score = 0.0

    # High centroid distance between benign and malicious = suspicious
    dist = analysis.get("centroid_distance")
    if dist is not None:
        # Distance > 0.3 is very suspicious, > 0.15 is moderate
        if dist > 0.3:
            score += 40.0
        elif dist > 0.15:
            score += 25.0
        elif dist > 0.05:
            score += 10.0

    # High intent-behavior distance = misalignment
    ib_dist = analysis.get("intent_behavior_distance")
    if ib_dist is not None:
        if ib_dist > 0.3:
            score += 35.0
        elif ib_dist > 0.15:
            score += 20.0
        elif ib_dist > 0.05:
            score += 10.0

    # Malicious content proportion
    total = analysis.get("total_events", 0)
    mal_count = analysis.get("malicious_count", 0)
    if total > 0:
        mal_ratio = mal_count / total
        if mal_ratio > 0.3:
            score += 25.0
        elif mal_ratio > 0.1:
            score += 15.0
        elif mal_ratio > 0.01:
            score += 5.0

    return min(score, 100.0)


def _run_scenario(
    scenario_fn: Any,
    api_key: str,
) -> dict[str, Any]:
    """Run a single scenario through all detection layers."""
    name, events, success_criteria = scenario_fn()
    print(f"\n  {name} ({len(events)} events)...")

    # Extract content for embedding
    contents = [e.content if e.content else "" for e in events]

    # Limit embedding calls for large scenarios (budget control)
    max_embed = 50
    if len(contents) > max_embed:
        # Sample: first 10, last 10, and 30 evenly spaced
        indices = list(range(10))
        step = max(1, (len(contents) - 20) // 30)
        indices += list(range(10, len(contents) - 10, step))[:30]
        indices += list(range(len(contents) - 10, len(contents)))
        indices = sorted(set(indices))
        sampled_contents = [contents[i] for i in indices]
        sampled_events = [events[i] for i in indices]
        print(f"    Sampling {len(indices)}/{len(contents)} events for embedding")
    else:
        sampled_contents = contents
        sampled_events = events
        indices = list(range(len(contents)))

    # Get embeddings from Gemini API
    embeddings = _embed_batch(sampled_contents, api_key)
    print(f"    Embedded {len(embeddings)} texts")

    # Embedding analysis (Layer 2)
    embedding_analysis = _analyze_scenario_embeddings(sampled_events, embeddings)

    # Layer 1 results (local analysis only)
    layer1_result = _run_with_config(events, "semantic_only")

    # Combined results (all brains)
    combined_result = _run_with_config(events, "combined")

    # Comparison
    layer1_detected = layer1_result["coefficient"] > 0.0
    layer2_detected = embedding_analysis["embedding_threat_score"] > 20.0
    combined_detected = combined_result["coefficient"] > 0.0

    return {
        "scenario": name,
        "success_criteria": success_criteria,
        "event_count": len(events),
        "embedded_count": len(sampled_contents),
        "layer1_local": {
            **layer1_result,
            "detected": layer1_detected,
        },
        "layer2_embedding": {
            **embedding_analysis,
            "detected": layer2_detected,
        },
        "combined_brains": {
            **combined_result,
            "detected": combined_detected,
        },
        "raw_embeddings": {
            "indices": indices,
            "vectors": embeddings,
        },
    }


def run_experiment() -> dict[str, Any]:
    """Run full live semantic detection experiment."""
    api_key = _load_api_key()

    scenarios = [
        _scenario_1_slow_drip,
        _scenario_2_synonym_bypass,
        _scenario_3_base64_injection,
        _scenario_4_security_researcher,
        _scenario_5_burst_compute,
        _scenario_7_micro_swarm,
        _scenario_8_threshold_surfing,
        _scenario_10_fan_out,
        _scenario_11_boiling_frog,
        _scenario_12_timestamp_manipulation,
    ]

    results: list[dict[str, Any]] = []
    for scenario_fn in scenarios:
        result = _run_scenario(scenario_fn, api_key)
        results.append(result)

    # Summary
    layer1_detected = sum(1 for r in results if r["layer1_local"]["detected"])
    layer2_detected = sum(1 for r in results if r["layer2_embedding"]["detected"])
    combined_detected = sum(1 for r in results if r["combined_brains"]["detected"])
    total = len(results)

    # Find scenarios with biggest embedding separation
    separations = []
    for r in results:
        dist = r["layer2_embedding"].get("centroid_distance")
        if dist is not None:
            separations.append((r["scenario"], dist))
    separations.sort(key=lambda x: x[1], reverse=True)

    summary = {
        "total_scenarios": total,
        "detection_rates": {
            "layer1_local": {
                "detected": layer1_detected,
                "total": total,
                "rate": round(layer1_detected / total, 2),
            },
            "layer2_embedding": {
                "detected": layer2_detected,
                "total": total,
                "rate": round(layer2_detected / total, 2),
            },
            "combined_brains": {
                "detected": combined_detected,
                "total": total,
                "rate": round(combined_detected / total, 2),
            },
        },
        "top_embedding_separations": [
            {"scenario": name, "centroid_distance": round(d, 4)}
            for name, d in separations[:5]
        ],
        "total_embedding_calls": sum(r["embedded_count"] for r in results),
    }

    return {
        "experiment": "semantic_live",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": _EMBED_MODEL,
        "scenarios": results,
        "summary": summary,
    }


def main() -> None:
    """Run experiment and save results."""
    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SybilCore Live Semantic Detection Experiment")
    print(f"Model: {_EMBED_MODEL}")
    print("=" * 60)

    start = time.time()
    results = run_experiment()
    elapsed = time.time() - start

    # Save results (strip raw embeddings for the summary print)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = _EXPERIMENTS_DIR / f"semantic_live_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"\nDetection Rates:")
    for config, stats in results["summary"]["detection_rates"].items():
        rate_pct = stats["rate"] * 100
        print(f"  {config:20s}: {stats['detected']}/{stats['total']} ({rate_pct:.0f}%)")

    print(f"\nTotal embedding API calls: {results['summary']['total_embedding_calls']}")

    print(f"\nTop Embedding Separations (malicious vs benign):")
    for item in results["summary"]["top_embedding_separations"]:
        print(f"  {item['scenario']:45s} distance={item['centroid_distance']:.4f}")

    print(f"\nScenario Details:")
    for r in results["scenarios"]:
        name = r["scenario"]
        l1 = "YES" if r["layer1_local"]["detected"] else "NO "
        l2 = "YES" if r["layer2_embedding"]["detected"] else "NO "
        cb = "YES" if r["combined_brains"]["detected"] else "NO "
        l2_score = r["layer2_embedding"]["embedding_threat_score"]
        l1_coeff = r["layer1_local"]["coefficient"]
        dist = r["layer2_embedding"].get("centroid_distance", "N/A")
        ib = r["layer2_embedding"].get("intent_behavior_distance", "N/A")

        print(f"\n  {name}:")
        print(f"    Layer 1 (local):    detected={l1}  coeff={l1_coeff:6.1f}")
        print(f"    Layer 2 (embed):    detected={l2}  score={l2_score:6.1f}  centroid_dist={dist}  intent_dist={ib}")
        print(f"    Combined (brains):  detected={cb}")


if __name__ == "__main__":
    main()
