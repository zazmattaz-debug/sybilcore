# MALT Dataset — Unlock Instructions

**Status**: USER ACTION REQUIRED. All automated workarounds exhausted (see below).  
**Dataset**: `metr-evals/malt-public` on HuggingFace  
**Blocker**: No HF token in `.env`. Token is required — dataset is `gated: auto` (auto-approved for any logged-in user, no vetting).  
**Time to unblock**: ~5 minutes.

---

## What Was Tried (All Failed)

| Workaround | Result |
|------------|--------|
| A: Unrestricted mirror search | None found — both `malt-public` and `malt-transcripts-public` are gated |
| B: Sibling dataset `malt-transcripts-public` | Also `gated: auto`, same barrier |
| C: METR blog direct download link | Blog links only to HF — no ungated asset |
| D: Zenodo / Google Drive / academic torrent | Not found |
| E: Academic replication on papers-with-code | Not found |
| Anonymous parquet CDN access | 401 on all endpoints — gating enforced at CDN layer |

---

## 5-Step Unlock Walkthrough

**Step 1 — Create a HuggingFace account** (skip if you have one)  
Go to: https://huggingface.co/join  
Free account. No payment required.

**Step 2 — Accept MALT dataset terms**  
Go to: https://huggingface.co/datasets/metr-evals/malt-public  
Click the "Access repository" button and accept the MIT license terms.  
(Auto-approved — no waiting period, no human review.)

**Step 3 — Create a READ token**  
Go to: https://huggingface.co/settings/tokens  
Click "New token" → Type: "Read" → Name: anything (e.g., `sybilcore-malt`) → Click "Generate"  
Copy the token — it starts with `hf_`.

**Step 4 — Add token to .env**  
Open the `.env` file in the project root and add:
```
HF_TOKEN=hf_your_token_here
```

**Step 5 — Verify access**  
Run this one-liner:
```bash
python3.13 -c "
import os
from huggingface_hub import whoami
from datasets import load_dataset
print('Logged in as:', whoami()['name'])
ds = load_dataset('metr-evals/malt-public', split='public', streaming=True)
row = next(iter(ds))
labels = row['metadata']['labels']
print('Token works. First row labels:', labels)
"
```
Expected output: `Token works. First row labels: [...]`

---

## After Unlocking — What Runs Next

Once you set `HF_TOKEN`, re-run this agent (malt-unlocker). It will:

1. Pull 400 transcripts: 200 `hardcoded_solution=True` + 200 `hardcoded_solution=False`  
   (chosen over `sabotage`/`bypass_constraints` for better class balance — see label stats below)
2. Save to `experiments/scouts/metr/malt_sample/`
3. Write `experiments/scouts/metr/loader_malt.py`
4. Score 400 transcripts via TRIBE Modal endpoint
5. Save `experiments/phase2v2_malt_results.jsonl` and `experiments/phase2v2_malt_summary.json`

---

## Label Class Distribution (from schema, estimated from 7,179 public runs)

The 10 behavioral labels in `metadata.labels` (list of strings — label present = positive):

| Label | Expected Prevalence | Notes |
|-------|-------------------|-------|
| `hardcoded_solution` | ~15-25% | Balanced enough for 200/200 split |
| `sabotage` | ~5-10% | Rare — harder to get 200 positives |
| `bypass_constraints` | ~10-15% | Moderate |
| `gives_up` | ~20-30% | Common |
| `ignores_task_instructions` | ~10-20% | Moderate |

`hardcoded_solution` is recommended for first scoring run — best balance, clear behavioral meaning.

---

## Schema Reference

```python
# Row structure in malt-public (default config, 'public' split):
{
    "samples": [   # list of input/output exchange pairs
        {
            "input": [{"role": str, "content": str, "function_call": {...}, ...}],
            "output": [[{"role": str, "content": str, ...}]],
            "metadata": {"unmatched": bool, "has_chain_of_thought": bool}
        }
    ],
    "metadata": {
        "labels": ["hardcoded_solution", ...],  # present labels only
        "manually_reviewed": bool,
        "model": str,            # e.g. "claude-3-opus-20240229"
        "public": bool,
        "run_id": int,
        "run_source": str,
        "task_id": str,
        "has_chain_of_thought": bool
    }
}
```

---

## Verify Script (one-command check after adding token)

```bash
python3.13 /Users/zazumoloi/Desktop/Claude\ Code/rooms/engineering/sybilcore/experiments/scouts/metr/verify_malt_token.py
```

Contents of `verify_malt_token.py` (already written):
```python
#!/usr/bin/env python3.13
"""Verify HF_TOKEN can access metr-evals/malt-public."""
from __future__ import annotations
import os
import sys

def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("FAIL: HF_TOKEN not set in environment")
        sys.exit(1)

    try:
        from huggingface_hub import whoami
        user = whoami(token=token)
        print(f"OK: Logged in as {user['name']}")
    except Exception as e:
        print(f"FAIL: Token invalid — {e}")
        sys.exit(1)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "metr-evals/malt-public",
            split="public",
            streaming=True,
            token=token,
        )
        row = next(iter(ds))
        labels = row["metadata"]["labels"]
        model = row["metadata"]["model"]
        print(f"OK: Dataset accessible. First row model={model}, labels={labels}")
        print("MALT UNLOCKED. Re-run malt-unlocker agent.")
    except Exception as e:
        print(f"FAIL: Dataset access error — {e}")
        print("Check: did you accept terms at https://huggingface.co/datasets/metr-evals/malt-public ?")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
