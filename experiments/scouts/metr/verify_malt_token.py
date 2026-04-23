#!/usr/bin/env python3.13
"""Verify HF_TOKEN can access metr-evals/malt-public."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path("/Users/zazumoloi/Desktop/Claude Code/.env"))


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
        print(f"OK: Dataset accessible. First row model={model!r}, labels={labels}")
        print("MALT UNLOCKED. Re-run malt-unlocker agent.")
    except Exception as e:
        print(f"FAIL: Dataset access error — {e}")
        print(
            "Check: did you accept terms at "
            "https://huggingface.co/datasets/metr-evals/malt-public ?"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
