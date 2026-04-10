"""Build and optionally upload the SybilCore agent-trust-benchmark-v1 dataset.

Sources:
  1. Moltbook live observatory (2,677 real agents) — most recent run.
     experiments/observatory_live_20260404_090758_full.jsonl
  2. Tournament v4 adversarial strategies (15,000 synthetic adversarial samples).
     experiments/tournament_20260407_161726_strategies.jsonl

Normalized schema (one record per agent / per adversarial iteration):
  {
    "agent_id":     str,            # unique within split
    "source":       str,            # "moltbook_live" | "tournament_v4_adversarial"
    "split":        str,            # "real_world" | "adversarial"
    "label":        str,            # "benign" | "likely_bot" | "adversarial_attack"
    "coefficient":  float,          # SybilCore trust coefficient (higher = more suspicious)
    "brain_scores": dict[str,float],# raw per-brain scores
    "events":       list[dict],     # event stream (may be empty for tournament rows)
    "metadata":     dict,           # source-specific extra fields
  }

Outputs:
  datasets/agent-trust-benchmark-v1/real_world.parquet
  datasets/agent-trust-benchmark-v1/adversarial.parquet
  datasets/agent-trust-benchmark-v1/metadata.json   (Croissant 1.0)
  datasets/agent-trust-benchmark-v1/README.md       (HuggingFace dataset card)

Usage:
  python3 scripts/build_hf_dataset.py             # build local files
  python3 scripts/build_hf_dataset.py --upload    # also upload to HuggingFace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUT_DIR = REPO_ROOT / "datasets" / "agent-trust-benchmark-v1"

MOLTBOOK_SRC = EXPERIMENTS_DIR / "observatory_live_20260404_090758_full.jsonl"
TOURNAMENT_SRC = EXPERIMENTS_DIR / "tournament_20260407_161726_strategies.jsonl"

DATASET_REPO_ID = "LordMoloi/sybilcore-agent-trust-v1"
DATASET_VERSION = "1.0.0"
LICENSE = "cc-by-4.0"


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_moltbook(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Moltbook observatory record into the common schema."""
    agent_id = record.get("agent_id") or "unknown"
    classification = record.get("classification") or "unknown"
    # Label mapping: any classification containing "bot" → likely_bot, else benign.
    label = "likely_bot" if "bot" in classification else "benign"

    brain_scores = {
        str(k): float(v) for k, v in (record.get("brain_scores") or {}).items()
    }

    metadata = {
        "tier": record.get("tier"),
        "classification": classification,
        "event_count": record.get("event_count"),
        "posting_cov": record.get("posting_cov"),
        "topic_entropy": record.get("topic_entropy"),
        "karma": record.get("karma"),
        "instruction_flags": record.get("instruction_flags"),
        "suspicion_score": record.get("suspicion_score"),
        "submolts": record.get("submolts") or [],
    }

    # The raw per-event stream is not stored in the observatory summary file.
    # We leave events empty but document this in the dataset card.
    events: list[dict[str, Any]] = []

    return {
        "agent_id": f"moltbook::{agent_id}",
        "source": "moltbook_live",
        "split": "real_world",
        "label": label,
        "coefficient": float(record.get("coefficient") or 0.0),
        "brain_scores_json": json.dumps(brain_scores, sort_keys=True),
        "events_json": json.dumps(events),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }


def normalize_tournament(record: dict[str, Any], row_idx: int) -> dict[str, Any]:
    """Normalize one tournament v4 adversarial strategy row."""
    model = record.get("model") or "unknown"
    iteration = int(record.get("iteration") or row_idx)
    # Stable, collision-free id combining model + iteration + content hash.
    h = hashlib.sha1(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    agent_id = f"tournament::{model}::{iteration:06d}::{h}"

    brain_scores = {
        str(k): float(v) for k, v in (record.get("brain_scores") or {}).items()
    }

    metadata = {
        "attacker_model": model,
        "iteration": iteration,
        "strategy": record.get("strategy"),
    }

    return {
        "agent_id": agent_id,
        "source": "tournament_v4_adversarial",
        "split": "adversarial",
        "label": "adversarial_attack",
        "coefficient": float(record.get("coefficient") or 0.0),
        "brain_scores_json": json.dumps(brain_scores, sort_keys=True),
        "events_json": json.dumps([]),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }


def write_parquet(rows: list[dict[str, Any]], out_path: Path) -> int:
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")
    table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    return len(rows)


def build_croissant_metadata(
    real_count: int,
    adversarial_count: int,
    real_parquet: str,
    adv_parquet: str,
) -> dict[str, Any]:
    """Build Croissant 1.0 metadata per mlcommons.org/croissant spec."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "name": "agent-trust-benchmark-v1",
        "description": (
            "SybilCore agent-trust-benchmark-v1: a dual-split benchmark for "
            "behavioral trust monitoring of AI agents. Combines 2,677 real "
            "Moltbook agents scored by the 15-brain SybilCore ensemble with "
            "15,000 adversarial iterations harvested from a 3-LLM tournament "
            "(Gemini Pro, GPT-4o, Grok). Each record contains per-brain "
            "scores, a trust coefficient, label, and source attribution."
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"https://huggingface.co/datasets/{DATASET_REPO_ID}",
        "version": DATASET_VERSION,
        "datePublished": now,
        "creator": {
            "@type": "sc:Organization",
            "name": "SybilCore Research",
            "url": "https://github.com/sybilcore/sybilcore",
        },
        "citeAs": (
            "@misc{sybilcore2026benchmark,\n"
            "  title={SybilCore Agent Trust Benchmark v1},\n"
            "  author={SybilCore Research},\n"
            "  year={2026},\n"
            "  howpublished={\\url{https://huggingface.co/datasets/"
            f"{DATASET_REPO_ID}}}\n"
            "}"
        ),
        "keywords": [
            "AI agents",
            "trust monitoring",
            "anomaly detection",
            "adversarial evaluation",
            "behavioral biometrics",
            "LLM security",
        ],
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "real_world.parquet",
                "name": "real_world.parquet",
                "contentUrl": real_parquet,
                "encodingFormat": "application/x-parquet",
                "description": (
                    f"{real_count} real-world Moltbook agents with SybilCore "
                    "trust scores and bot classifications."
                ),
            },
            {
                "@type": "cr:FileObject",
                "@id": "adversarial.parquet",
                "name": "adversarial.parquet",
                "contentUrl": adv_parquet,
                "encodingFormat": "application/x-parquet",
                "description": (
                    f"{adversarial_count} adversarial iterations harvested "
                    "from a 15k-round tournament across 3 LLMs."
                ),
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "real_world",
                "name": "real_world",
                "description": "Moltbook real-world agents split.",
                "field": _common_fields("real_world.parquet"),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "adversarial",
                "name": "adversarial",
                "description": "Tournament v4 adversarial iterations split.",
                "field": _common_fields("adversarial.parquet"),
            },
        ],
    }


def _common_fields(file_obj_id: str) -> list[dict[str, Any]]:
    def field(name: str, dtype: str, desc: str) -> dict[str, Any]:
        return {
            "@type": "cr:Field",
            "@id": f"{file_obj_id}/{name}",
            "name": name,
            "description": desc,
            "dataType": dtype,
            "source": {
                "fileObject": {"@id": file_obj_id},
                "extract": {"column": name},
            },
        }

    return [
        field("agent_id", "sc:Text", "Unique agent identifier prefixed by source."),
        field("source", "sc:Text", "Source system: moltbook_live or tournament_v4_adversarial."),
        field("split", "sc:Text", "Split name: real_world or adversarial."),
        field("label", "sc:Text", "benign | likely_bot | adversarial_attack."),
        field("coefficient", "sc:Float", "SybilCore trust coefficient (higher = more suspicious)."),
        field("brain_scores_json", "sc:Text", "JSON-encoded per-brain score dict."),
        field("events_json", "sc:Text", "JSON-encoded event stream (may be empty)."),
        field("metadata_json", "sc:Text", "JSON-encoded source-specific metadata."),
    ]


DATASET_CARD = """---
license: cc-by-4.0
task_categories:
- text-classification
- other
language:
- en
tags:
- ai-safety
- ai-agents
- trust-monitoring
- anomaly-detection
- adversarial-robustness
- behavioral-biometrics
- llm-security
pretty_name: SybilCore Agent Trust Benchmark v1
size_categories:
- 10K<n<100K
configs:
- config_name: real_world
  data_files:
  - split: train
    path: real_world.parquet
- config_name: adversarial
  data_files:
  - split: train
    path: adversarial.parquet
---

# SybilCore Agent Trust Benchmark v1

A dual-split benchmark for behavioral trust monitoring of AI agents, designed to
support research on trust, anomaly detection, and adversarial robustness of
LLM-backed agent systems.

## Overview

This dataset bundles two complementary splits scored by the 15-brain SybilCore
ensemble:

| Split         | Records | Source                                     | Label space                     |
|---------------|--------:|--------------------------------------------|---------------------------------|
| `real_world`  |   2,677 | Moltbook live observatory (April 2026)     | `benign`, `likely_bot`          |
| `adversarial` |  15,000 | Tournament v4 (Gemini Pro / GPT-4o / Grok) | `adversarial_attack`            |

## Schema

Each row contains:

- `agent_id` *(str)* — stable, source-prefixed identifier
- `source` *(str)* — `moltbook_live` or `tournament_v4_adversarial`
- `split` *(str)* — `real_world` or `adversarial`
- `label` *(str)* — `benign`, `likely_bot`, or `adversarial_attack`
- `coefficient` *(float)* — SybilCore trust coefficient (higher = more suspicious)
- `brain_scores_json` *(str)* — JSON dict of per-brain scores
- `events_json` *(str)* — JSON-encoded event stream (empty for summary splits)
- `metadata_json` *(str)* — JSON dict of source-specific extra fields

JSON-encoded columns are used so that the schema is stable across Parquet
readers; decode them at load time, e.g.:

```python
import json, pandas as pd
df = pd.read_parquet("real_world.parquet")
df["brain_scores"] = df["brain_scores_json"].map(json.loads)
```

## Splits in detail

### `real_world` — 2,677 Moltbook agents

Collected from a live social-agent experiment (Moltbook, a Reddit-style
simulation). Each agent has a trust coefficient and per-brain scores computed
by the SybilCore ensemble, plus the original platform-level signals
(posting regularity CoV, topic entropy, karma, instruction-flag counts, and a
bot classification). The `events_json` column is empty for this split: the
observatory file ships pre-aggregated scores rather than the raw event stream.
Raw per-event data can be regenerated from the SybilCore Moltbook harness.

### `adversarial` — 15,000 tournament iterations

Harvested from a 15,000-iteration adversarial tournament in which three LLM
attackers (Gemini Pro, GPT-4o, Grok) each produced 5,000 candidate evasion
strategies against the 15-brain SybilCore scorer. The most striking finding
is near-total convergence: 98–99.6% of the winning strategies across all
three models collapse onto `multi_channel_exfiltration`. Per-iteration
metadata stores the attacker model, iteration number, and the strategy label.

## Provenance and labels

- Real-world `likely_bot` labels come from the Moltbook observatory's
  heuristic bot classifier, not from human annotation. They should be treated
  as weak labels rather than ground truth.
- Adversarial rows are labeled `adversarial_attack` by construction — every
  row is the output of an optimizer searching for high-coefficient behavior.

## Intended use

- Benchmarking trust / anomaly detectors for AI agent behavior
- Studying adversarial robustness and cross-model attack convergence
- Ablation and calibration of multi-signal trust scoring systems
- Reproducibility baseline for the SybilCore NeurIPS 2026 submission

## Limitations

- The real-world split's labels are heuristic, not human-reviewed.
- The adversarial split does not contain the raw event stream — only
  per-brain scores, coefficient, and strategy metadata are preserved.
- The benchmark reflects one ensemble architecture (SybilCore v4); detectors
  trained only on this distribution may not transfer to unrelated systems.

## Citation

```bibtex
@misc{sybilcore2026benchmark,
  title   = {SybilCore Agent Trust Benchmark v1},
  author  = {SybilCore Research},
  year    = {2026},
  howpublished = {\\url{https://huggingface.co/datasets/LordMoloi/sybilcore-agent-trust-v1}}
}
```

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""


def load_env_token() -> str | None:
    """Read HF_TOKEN from environment or workspace .env without printing it."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    # Fallback: parse the workspace .env so the script works outside a shell
    # session that exported the variable.
    candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parents[1] / ".env",  # rooms/.env (unlikely)
        REPO_ROOT.parents[2] / ".env",  # workspace root (Claude Code/.env)
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            with env_path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip() == "HF_TOKEN":
                        return v.strip().strip('"').strip("'")
        except OSError:
            continue
    return None


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not MOLTBOOK_SRC.exists():
        raise FileNotFoundError(f"Moltbook source missing: {MOLTBOOK_SRC}")
    if not TOURNAMENT_SRC.exists():
        raise FileNotFoundError(f"Tournament source missing: {TOURNAMENT_SRC}")

    print(f"[build] Reading Moltbook agents from {MOLTBOOK_SRC.name}")
    real_rows = [normalize_moltbook(r) for r in load_jsonl(MOLTBOOK_SRC)]
    print(f"[build]   -> {len(real_rows)} real-world agents")

    print(f"[build] Reading tournament strategies from {TOURNAMENT_SRC.name}")
    adv_rows = [
        normalize_tournament(r, i)
        for i, r in enumerate(load_jsonl(TOURNAMENT_SRC))
    ]
    print(f"[build]   -> {len(adv_rows)} adversarial iterations")

    real_path = OUT_DIR / "real_world.parquet"
    adv_path = OUT_DIR / "adversarial.parquet"
    write_parquet(real_rows, real_path)
    write_parquet(adv_rows, adv_path)
    print(f"[build] Wrote {real_path.relative_to(REPO_ROOT)}")
    print(f"[build] Wrote {adv_path.relative_to(REPO_ROOT)}")

    croissant = build_croissant_metadata(
        real_count=len(real_rows),
        adversarial_count=len(adv_rows),
        real_parquet="real_world.parquet",
        adv_parquet="adversarial.parquet",
    )
    metadata_path = OUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(croissant, indent=2), encoding="utf-8")
    print(f"[build] Wrote {metadata_path.relative_to(REPO_ROOT)} (Croissant 1.0)")

    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(DATASET_CARD, encoding="utf-8")
    print(f"[build] Wrote {readme_path.relative_to(REPO_ROOT)}")

    return {
        "real_count": len(real_rows),
        "adversarial_count": len(adv_rows),
        "real_path": real_path,
        "adv_path": adv_path,
        "metadata_path": metadata_path,
        "readme_path": readme_path,
    }


def upload(summary: dict[str, Any]) -> str | None:
    token = load_env_token()
    if not token:
        print(
            "[upload] HF_TOKEN not found in environment or .env; skipping upload.",
            file=sys.stderr,
        )
        return None

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:  # pragma: no cover
        print(f"[upload] huggingface_hub not available: {exc}", file=sys.stderr)
        return None

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
    except HfHubHTTPError as exc:
        print(
            f"[upload] Failed to create/access dataset repo {DATASET_REPO_ID}: {exc}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:  # pragma: no cover
        print(f"[upload] Unexpected error creating repo: {exc}", file=sys.stderr)
        return None

    try:
        api.upload_folder(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            folder_path=str(OUT_DIR),
            commit_message=f"Upload agent-trust-benchmark-v{DATASET_VERSION}",
        )
    except HfHubHTTPError as exc:
        print(f"[upload] Upload failed: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # pragma: no cover
        print(f"[upload] Unexpected error during upload: {exc}", file=sys.stderr)
        return None

    url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
    print(f"[upload] Success: {url}")
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the built dataset to HuggingFace after writing local files.",
    )
    args = parser.parse_args()

    summary = build(args)
    print(
        f"[build] Done. real_world={summary['real_count']}, "
        f"adversarial={summary['adversarial_count']}"
    )

    if args.upload:
        url = upload(summary)
        if url:
            print(f"[upload] Dataset URL: {url}")
        else:
            print("[upload] Upload blocked or failed — local files are intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
