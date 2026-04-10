"""FastAPI web UI for the SybilCore eval corpus.

Run with:
    python -m sybilcore.evaluation.web_ui --corpus experiments/human_eval_corpus_v4.json

Then open http://localhost:8765 in a browser. Multiple raters can label
concurrently — the rater_id is taken from a query string and stored
alongside each judgment in a SQLite database.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from sybilcore.evaluation.human_eval import (
    AgentRecord,
    HumanEvalFramework,
)

logger = logging.getLogger(__name__)


# ── Storage ─────────────────────────────────────────────────────────


class LabelStore:
    """Lightweight SQLite store for multi-rater judgments."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judgments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    rater_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(agent_id, rater_id)
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def upsert(
        self,
        agent_id: str,
        rater_id: str,
        label: str,
        confidence: int,
        notes: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO judgments (agent_id, rater_id, label, confidence, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, rater_id) DO UPDATE SET
                    label = excluded.label,
                    confidence = excluded.confidence,
                    notes = excluded.notes,
                    created_at = excluded.created_at
                """,
                (agent_id, rater_id, label, int(confidence), notes, datetime.now(UTC).isoformat()),
            )
            conn.commit()

    def labeled_agent_ids(self, rater_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_id FROM judgments WHERE rater_id = ?", (rater_id,)
            ).fetchall()
            return {r["agent_id"] for r in rows}

    def export(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM judgments ORDER BY id").fetchall()
            return [dict(r) for r in rows]


# ── App factory ─────────────────────────────────────────────────────


def create_app(corpus_path: Path, db_path: Path) -> FastAPI:
    """Build a FastAPI app bound to a corpus + label database."""
    records = HumanEvalFramework.load_corpus(corpus_path)
    record_index = {r.agent_id: r for r in records}
    store = LabelStore(db_path)

    app = FastAPI(title="SybilCore Human Eval", version="1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:  # pragma: no cover - trivial template
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/progress")
    def api_progress(rater: str = Query("default")) -> dict[str, Any]:
        labeled = store.labeled_agent_ids(rater)
        return {
            "rater": rater,
            "total": len(records),
            "labeled": len(labeled),
            "remaining": len(records) - len(labeled),
        }

    @app.get("/api/next")
    def api_next(rater: str = Query("default")) -> dict[str, Any]:
        labeled = store.labeled_agent_ids(rater)
        for r in records:
            if r.agent_id in labeled:
                continue
            return {
                "agent": _record_to_view(r),
                "formatted": HumanEvalFramework.format_for_review(r),
                "progress": {
                    "labeled": len(labeled),
                    "total": len(records),
                },
            }
        return {"agent": None, "progress": {"labeled": len(labeled), "total": len(records)}}

    @app.post("/api/label")
    def api_label(payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", "")).strip()
        if agent_id not in record_index:
            raise HTTPException(status_code=404, detail="agent_id not in corpus")
        label = str(payload.get("label", "")).strip()
        if label not in {"suspicious", "benign", "skip"}:
            raise HTTPException(status_code=400, detail="invalid label")
        try:
            confidence = int(payload.get("confidence", 3))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="confidence must be int") from exc
        if not 1 <= confidence <= 5:
            raise HTTPException(status_code=400, detail="confidence out of range")
        notes = str(payload.get("notes", ""))
        rater = str(payload.get("rater", "default"))
        store.upsert(agent_id, rater, label, confidence, notes)
        return {"status": "ok"}

    @app.get("/api/export")
    def api_export() -> JSONResponse:
        return JSONResponse(content=store.export())

    return app


def _record_to_view(r: AgentRecord) -> dict[str, Any]:
    return {
        "agent_id": r.agent_id,
        "source": r.source,
        "ground_truth": r.ground_truth,
        "sybilcore_coefficient": r.sybilcore_coefficient,
        "sybilcore_tier": r.sybilcore_tier,
        "sybilcore_suspicious": r.sybilcore_suspicious,
        "brain_scores": r.brain_scores,
        "summary": r.summary,
        "events": r.events[:12],
    }


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SybilCore Human Eval</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 760px; margin: 32px auto; padding: 0 16px; color: #111; }
  h1 { font-size: 22px; }
  pre { background: #0e0e0e; color: #b8e986; padding: 16px; border-radius: 8px; overflow: auto; font-size: 12px; }
  .row { margin: 12px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { padding: 10px 14px; font-size: 14px; border: 1px solid #ccc; background: #fafafa; border-radius: 6px; cursor: pointer; }
  button.danger { background: #ffe5e5; border-color: #f88; }
  button.safe { background: #e7f7e7; border-color: #6c6; }
  input, textarea { padding: 8px; border: 1px solid #ccc; border-radius: 6px; font: inherit; }
  .progress { background: #eee; border-radius: 8px; height: 10px; overflow: hidden; }
  .progress > div { background: #4a90e2; height: 100%; transition: width 0.3s; }
  .meta { color: #555; font-size: 12px; }
</style>
</head>
<body>
<h1>SybilCore Human Eval</h1>
<div class="row">
  <label>Rater ID: <input id="rater" value="default"></label>
  <button onclick="loadNext()">Load next</button>
  <button onclick="exportLabels()">Export JSON</button>
</div>
<div class="row meta">
  <span id="progress-text">progress: …</span>
  <div class="progress" style="flex:1;"><div id="bar" style="width:0%"></div></div>
</div>
<pre id="agent">Click "Load next" to begin labeling.</pre>
<div class="row">
  <button class="danger" onclick="submit('suspicious')">Suspicious (y)</button>
  <button class="safe" onclick="submit('benign')">Benign (n)</button>
  <button onclick="submit('skip')">Skip</button>
  <label>Confidence: <input id="conf" type="number" min="1" max="5" value="3" style="width:60px"></label>
</div>
<div class="row">
  <textarea id="notes" rows="2" style="width:100%" placeholder="Notes (optional)"></textarea>
</div>
<script>
let current = null;
async function loadNext() {
  const rater = document.getElementById('rater').value || 'default';
  const r = await fetch('/api/next?rater=' + encodeURIComponent(rater));
  const data = await r.json();
  current = data.agent;
  if (!current) {
    document.getElementById('agent').textContent = 'Done — all agents labeled.';
  } else {
    document.getElementById('agent').textContent = data.formatted;
  }
  document.getElementById('notes').value = '';
  updateProgress(data.progress);
}
function updateProgress(p) {
  if (!p) return;
  const pct = p.total ? Math.round(100 * p.labeled / p.total) : 0;
  document.getElementById('bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent =
    'progress: ' + p.labeled + ' / ' + p.total + ' (' + pct + '%)';
}
async function submit(label) {
  if (!current) { return; }
  const rater = document.getElementById('rater').value || 'default';
  const conf = parseInt(document.getElementById('conf').value || '3', 10);
  const notes = document.getElementById('notes').value || '';
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      agent_id: current.agent_id, rater, label, confidence: conf, notes,
    }),
  });
  loadNext();
}
async function exportLabels() {
  const r = await fetch('/api/export');
  const data = await r.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'human_eval_labels.json';
  a.click();
  URL.revokeObjectURL(url);
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'y') submit('suspicious');
  else if (e.key === 'n') submit('benign');
  else if (e.key === 's') submit('skip');
});
loadNext();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SybilCore human eval web UI")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("experiments/human_eval_labels.db"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    import uvicorn  # imported lazily so tests don't require uvicorn

    app = create_app(args.corpus, args.db)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
