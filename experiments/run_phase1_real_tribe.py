#!/usr/bin/env python3
"""Phase 1 — Real TRIBE v2 Modal scoring. Run overnight or when endpoint is warm."""
import json, sys, time, urllib.request
from pathlib import Path
from collections import defaultdict

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
DATA_DIR = Path(__file__).parent / "phase1_data/cornell_deception/op_spam_v1.4/positive_polarity"
OUT_FILE = Path(__file__).parent / "phase1_cornell_real_tribe_v2_results.json"

def score_text(text):
    payload = json.dumps({"text": text, "modality": "text"}).encode()
    req = urllib.request.Request(SCORE_URL, data=payload, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return 0.5
    return (sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)) / (len(pos) * len(neg))

reviews = []
for label_dir, label in [("deceptive_from_MTurk", 1), ("truthful_from_TripAdvisor", 0)]:
    for fold_num in range(1, 6):
        fold_dir = DATA_DIR / label_dir / f"fold{fold_num}"
        if not fold_dir.exists(): continue
        for f in sorted(fold_dir.glob("*.txt")):
            reviews.append({"text": f.read_text(errors="replace").strip(), "label": label, "fold": fold_num})

print(f"Loaded {len(reviews)} reviews", flush=True)
results, errors = [], 0
t0 = time.time()

for i, rev in enumerate(reviews):
    try:
        resp = score_text(rev["text"])
        results.append({"label": rev["label"], "fold": rev["fold"],
            "emotion_score": resp.get("emotion_score", 0), "emotion_ratio": resp.get("emotion_ratio", 0)})
    except Exception as e:
        errors += 1
        if errors <= 5: print(f"  Err {errors}: {e}", flush=True)
        if errors > 50: break
        time.sleep(3)
    if (i+1) % 50 == 0:
        rate = (i+1) / (time.time() - t0)
        print(f"  {i+1}/{len(reviews)} ({rate:.1f}/s, ETA {(len(reviews)-i-1)/rate:.0f}s, {errors} err)", flush=True)

print(f"\nDone: {len(results)}/{len(reviews)} in {time.time()-t0:.0f}s", flush=True)

fd = defaultdict(lambda: {"es": [], "er": [], "l": []})
for r in results:
    fd[r["fold"]]["es"].append(r["emotion_score"])
    fd[r["fold"]]["er"].append(r["emotion_ratio"])
    fd[r["fold"]]["l"].append(r["label"])

for f in sorted(fd):
    print(f"  Fold {f}: es={auroc(fd[f]['es'],fd[f]['l']):.4f} er={auroc(fd[f]['er'],fd[f]['l']):.4f}", flush=True)
es_m = sum(auroc(fd[f]["es"],fd[f]["l"]) for f in fd)/len(fd)
er_m = sum(auroc(fd[f]["er"],fd[f]["l"]) for f in fd)/len(fd)
print(f"\nemotion_score AUROC={es_m:.4f}  emotion_ratio AUROC={er_m:.4f}  PASS={max(es_m,er_m)>0.85}", flush=True)
json.dump({"model": "TRIBE v2 Modal (A10G)", "es_auroc": es_m, "er_auroc": er_m, "n": len(results), "errors": errors}, open(str(OUT_FILE), "w"), indent=2)
