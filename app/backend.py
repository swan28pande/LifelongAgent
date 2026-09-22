"""
API for the memory inspector UI.

    python3 app/backend.py                                  # newest run under results/
    STORE=results/v3_pilot/store python3 app/backend.py     # pin a specific one

With no STORE set it opens the most recently written store under results/, which is
almost always the run you just finished. Every run is still listed on /api/status, so
the UI can show what else is on disk.

Read-only over whatever store it opens, plus a live chat endpoint. The store is opened
lazily so the server starts instantly instead of waiting on the embedding model, and
so an empty results/ gives a clear error in the UI rather than a crash at import time.
"""

import json
import os
import sys
from typing import Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memory_v3.agent import AgenticMemoryAgent  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def discover_stores() -> list:
    """Every run under results/ that has a store, newest first."""
    found = []
    if os.path.isdir(RESULTS_DIR):
        for name in os.listdir(RESULTS_DIR):
            db = os.path.join(RESULTS_DIR, name, "store", "memories.db")
            if os.path.exists(db):
                found.append({
                    "run": name,
                    "store": os.path.join(RESULTS_DIR, name, "store"),
                    "modified": os.path.getmtime(db),
                })
    return sorted(found, key=lambda s: -s["modified"])


def resolve_store() -> Optional[str]:
    """STORE wins if set; otherwise the most recently written run."""
    pinned = os.getenv("STORE")
    if pinned:
        return os.path.join(PROJECT_ROOT, pinned)
    stores = discover_stores()
    return stores[0]["store"] if stores else None


STORE_DIR = resolve_store()
DATASET = os.path.join(
    PROJECT_ROOT, os.getenv("DATASET", "datasets/eval/conversations.json")
)
QA_PATH = os.path.join(PROJECT_ROOT, "datasets", "eval", "qa_pairs.json")
LOCOMO_PATH = os.path.join(PROJECT_ROOT, "baselines", "locomo", "data", "locomo10.json")
MODEL = os.getenv("MODEL", "gemini-3.5-flash")

app = FastAPI(title="Lifelong Agent Inspector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Optional[AgenticMemoryAgent] = None


def agent() -> AgenticMemoryAgent:
    """Open the store on first use — loading embeddings takes ~30s."""
    global _agent
    if _agent is None:
        if not STORE_DIR or not os.path.exists(os.path.join(STORE_DIR, "memories.db")):
            raise HTTPException(
                status_code=404,
                detail="No store found under results/. Run scripts/run_memory_v3.py "
                       "first, or set STORE=<path> when starting the server.",
            )
        _agent = AgenticMemoryAgent(base_dir=STORE_DIR, model=MODEL)
    return _agent


class ChatRequest(BaseModel):
    message: str


# ── Status ──────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """What this server opened, and what else is available on disk."""
    has_store = bool(STORE_DIR) and os.path.exists(
        os.path.join(STORE_DIR, "memories.db")
    )
    body = {
        "system": "memory_v3",
        "model": MODEL,
        "run": os.path.basename(os.path.dirname(STORE_DIR)) if STORE_DIR else None,
        "store": os.path.relpath(STORE_DIR, PROJECT_ROOT) if STORE_DIR else None,
        "pinned": bool(os.getenv("STORE")),
        "dataset": os.path.relpath(DATASET, PROJECT_ROOT),
        "store_exists": has_store,
        "loaded": _agent is not None,
        "available_runs": [s["run"] for s in discover_stores()],
    }
    if has_store and _agent is not None:
        start, end = _agent.store.get_date_range()
        body.update(date_range=[start, end])
    return body


@app.get("/api/stats")
def stats():
    a = agent()
    rows = a.store.query_memories(limit=10000)
    start, end = a.store.get_date_range()

    per_entity: Dict[str, int] = {}
    for r in rows:
        per_entity[r["entity"]] = per_entity.get(r["entity"], 0) + 1

    n_summaries = 0
    if a.store._summary_store:
        n_summaries = len(a.store._summary_store.docstore._dict)

    with a.store._conn() as conn:
        n_chunks = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

    return {
        "preferences": len(rows),
        "entities": sorted(per_entity.items(), key=lambda kv: -kv[1]),
        "chunks": n_chunks,
        "summaries": n_summaries,
        "date_range": [start, end],
        "speakers": a.store.get_all_speakers(),
    }


# ── Dataset ─────────────────────────────────────────────────────────

@app.get("/api/dataset")
def dataset():
    """
    The source conversations, newest field names normalised for the UI.

    Handles both dataset shapes: the eval set keyed by user with `sessions`, and the
    legacy per-day files with `interactions`.
    """
    if not os.path.exists(DATASET):
        raise HTTPException(status_code=404, detail=f"No dataset at {DATASET}")

    with open(DATASET) as f:
        raw = json.load(f)

    days = []

    if "user_1" in raw and "sessions" in raw["user_1"]:
        for date, session in sorted(raw["user_1"]["sessions"].items()):
            days.append({
                "date": date,
                "time_of_day": session.get("time_of_day", "All Day"),
                "turns": session.get("turns", []),
            })
    else:
        for date, info in sorted(raw.items()):
            for block in info.get("interactions", info.get("conversations", [])):
                days.append({
                    "date": date,
                    "time_of_day": block.get("time_of_day", ""),
                    "turns": block.get("turns", []),
                })

    return {"days": days, "count": len(days)}


@app.get("/api/qa")
def qa_pairs():
    """The evaluation questions, plus scored results if a run produced them."""
    if not os.path.exists(QA_PATH):
        raise HTTPException(status_code=404, detail="No qa_pairs.json")
    with open(QA_PATH) as f:
        pairs = json.load(f)["user_1"]["qa_pairs"]

    results_path = os.path.join(os.path.dirname(STORE_DIR), "qa_results.json")
    scored, summary = {}, None
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        summary = data.get("summary")
        scored = {r["question"]: r for r in data.get("records", [])}

    return {
        "summary": summary,
        "pairs": [{**p, **{k: v for k, v in scored.get(p["question"], {}).items()
                           if k in ("response", "f1", "llm")}}
                  for p in pairs],
    }


# ── Memory store ────────────────────────────────────────────────────

@app.get("/api/memories")
def memories(entity: Optional[str] = None, limit: int = 2000):
    return agent().store.query_memories(entity=entity, limit=limit)


@app.get("/api/entities")
def entities():
    return agent().store.get_all_entities()


@app.get("/api/chunks")
def chunks(limit: int = 200):
    """Raw conversation chunks held in the RAG store."""
    docs = agent().store.get_recent_conversations(k=limit)
    return [
        {"content": d.page_content,
         "date": d.metadata.get("source_date", ""),
         "speaker": d.metadata.get("speaker", "")}
        for d in docs
    ]


@app.get("/api/summaries")
def summaries():
    a = agent()
    if not a.store._summary_store:
        return []
    out = []
    for doc in a.store._summary_store.docstore._dict.values():
        meta = doc.metadata or {}
        identifier = meta.get("identifier", "")
        out.append({
            "identifier": identifier,
            "level": identifier.split(":")[0] if identifier else "",
            "title": meta.get("title", ""),
            "speaker": meta.get("speaker", ""),
            "content": doc.page_content,
        })
    return sorted(out, key=lambda d: d["identifier"])


# ── Chat ────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(request: ChatRequest):
    a = agent()
    try:
        return {"response": a.chat(request.message)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── LoCoMo ─────────────────────────────────────────────────────────

import re as _re

def _locomo_results_dir() -> Optional[str]:
    """Most recent locomo results directory."""
    candidates = []
    for name in os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []:
        path = os.path.join(RESULTS_DIR, name, "locomo_results.json")
        if os.path.exists(path):
            candidates.append((os.path.getmtime(path), os.path.join(RESULTS_DIR, name)))
    return max(candidates)[1] if candidates else None


@app.get("/api/locomo/dataset")
def locomo_dataset(conversation: Optional[int] = None):
    """LoCoMo source conversations."""
    if not os.path.exists(LOCOMO_PATH):
        raise HTTPException(status_code=404, detail="locomo10.json not found")
    with open(LOCOMO_PATH) as f:
        convs = json.load(f)

    result = []
    for i, conv in enumerate(convs):
        c = conv["conversation"]
        sessions = []
        keys = sorted(
            (k for k in c if _re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        for key in keys:
            date_raw = c.get(f"{key}_date_time", "")
            turns = [
                {"speaker": t["speaker"], "text": t.get("text", "")}
                for t in c[key] if "text" in t
            ]
            sessions.append({"key": key, "date_raw": date_raw, "turns": turns})

        entry = {
            "index": i,
            "sample_id": conv.get("sample_id", f"conv-{i}"),
            "speaker_a": c.get("speaker_a", ""),
            "speaker_b": c.get("speaker_b", ""),
            "sessions": sessions,
            "n_questions": len(conv.get("qa", [])),
        }
        result.append(entry)

    if conversation is not None and 0 <= conversation < len(result):
        return result[conversation]
    return {"conversations": result, "count": len(result)}


@app.get("/api/locomo/qa")
def locomo_qa():
    """LoCoMo QA results from the most recent run."""
    results_dir = _locomo_results_dir()
    if not results_dir:
        raise HTTPException(status_code=404, detail="No LoCoMo results found")
    path = os.path.join(results_dir, "locomo_results.json")
    with open(path) as f:
        data = json.load(f)
    return {
        "run": os.path.basename(results_dir),
        "model": data.get("model", ""),
        "summary": data.get("summary", {}),
        "records": data.get("records", []),
    }


if __name__ == "__main__":
    if STORE_DIR:
        how = "pinned by STORE" if os.getenv("STORE") else "newest under results/"
        print(f"store:   {os.path.relpath(STORE_DIR, PROJECT_ROOT)}  ({how})")
        others = [s["run"] for s in discover_stores()][1:]
        if others and not os.getenv("STORE"):
            print(f"         also available: {', '.join(others)}")
    else:
        print("store:   none found under results/ — run scripts/run_memory_v3.py")
    print(f"dataset: {os.path.relpath(DATASET, PROJECT_ROOT)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
