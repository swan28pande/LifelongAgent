"""
mem0 (open-source, v2.x) on LoCoMo, single-pass read path — the external baseline.

    venv/bin/python benchmarks/locomo/run_mem0.py --conversations 1

Held equal to the v3 runs so the comparison isolates the memory system: same LLM
(Gemini on Vertex), same embedding model (nomic-embed-text-v1), same dataset parsing,
same scoring, and the same answer prompt as run_v3_onepass.py.

Each session is added to mem0 in date order (mem0 extracts facts itself); each question
retrieves the top-k memories and one LLM call answers from them.

Output: results/locomo/<run>/ — locomo_results.json, <sample_id>_trace.jsonl (one line
per answer, with the retrieved memories), stores/<sample_id>/.
"""

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage, SystemMessage
from mem0 import Memory
from tqdm import tqdm

from benchmarks.locomo.common import (ANSWER_SYSTEM, RESULTS_ROOT, append_trace,
                                      get_sessions, load_dataset, make_record,
                                      print_report, save_results, select_questions,
                                      session_to_turns, summarize, text_of)
from memory_v3.agent import _make_llm, _vertex_project

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBED_DIMS = 768


def make_memory(store_dir: str, model: str) -> Memory:
    return Memory.from_config({
        "llm": {"provider": "gemini", "config": {
            # mem0's default of 2000 includes Gemini's thinking tokens, which truncates
            # the extraction JSON mid-string and silently yields no memories.
            "model": model, "temperature": 0.0, "max_tokens": 32000, "vertexai": True,
            "project": _vertex_project(),
            "location": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        }},
        "embedder": {"provider": "huggingface", "config": {
            "model": EMBED_MODEL, "embedding_dims": EMBED_DIMS,
            "model_kwargs": {"trust_remote_code": True},
        }},
        "vector_store": {"provider": "faiss", "config": {
            "path": store_dir, "collection_name": "locomo",
            "embedding_model_dims": EMBED_DIMS,
        }},
        "history_db_path": os.path.join(store_dir, "history.db"),
    })


def ingest_conversation(memory: Memory, conv: dict, user_id: str) -> dict:
    """Add every session in date order. Names and dates go in the text so mem0 keeps them."""
    stats = {"sessions": 0, "memories_added": 0, "errors": []}
    for date, turns in get_sessions(conv):
        formatted = session_to_turns(turns)
        if not formatted:
            continue
        messages = [{"role": "user", "content": f"[{date}] {t['speaker']}: {t['text']}"}
                    for t in formatted]
        try:
            result = memory.add(messages, user_id=user_id, metadata={"date": date})
            added = len(result.get("results", []))
            stats["memories_added"] += added
            # mem0 swallows extraction parse failures, so an empty session is the only signal.
            if added == 0:
                stats["errors"].append(f"{date}: no memories extracted")
            print(f"  [{date}] +{added} memories")
        except Exception as e:
            stats["errors"].append(f"{date}: {e}")
        stats["sessions"] += 1
    return stats


def answer(memory: Memory, llm, question: str, user_id: str, top_k: int) -> dict:
    hits = memory.search(question, filters={"user_id": user_id}, top_k=top_k)
    hits = hits.get("results", []) if isinstance(hits, dict) else hits
    retrieved = [{"memory": h["memory"], "date": (h.get("metadata") or {}).get("date"),
                  "score": h.get("score")} for h in hits]
    context = "\n".join(
        f"- [{r['date']}] {r['memory']}" if r["date"] else f"- {r['memory']}"
        for r in retrieved
    ) or "(no memories)"
    reply = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"MEMORY CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ])
    return {"answer": text_of(reply.content), "retrieved": retrieved,
            "context_chars": len(context)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--run", default="mem0")
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run)
    os.makedirs(run_dir, exist_ok=True)

    llm = _make_llm(args.model, 0.0)
    records, ingest_summary = [], []

    for i, conv in enumerate(load_dataset(args.conversations), 1):
        sample_id = conv.get("sample_id", f"conv{i}")
        print(f"\n{'=' * 64}\n[{i}] {sample_id}")

        store_dir = os.path.join(run_dir, "stores", sample_id)
        shutil.rmtree(store_dir, ignore_errors=True)
        os.makedirs(store_dir, exist_ok=True)
        memory = make_memory(store_dir, args.model)

        t0 = time.time()
        stats = ingest_conversation(memory, conv, user_id=sample_id)
        stats.update(sample_id=sample_id, seconds=round(time.time() - t0, 1),
                     total_memories=len(memory.get_all(filters={"user_id": sample_id},
                                                       top_k=100000).get("results", [])))
        ingest_summary.append(stats)
        print(f"  ingested {stats['sessions']} sessions → {stats['total_memories']} memories, "
              f"{stats['seconds']}s, {len(stats['errors'])} errors")

        trace_path = os.path.join(run_dir, f"{sample_id}_trace.jsonl")
        open(trace_path, "w").close()
        for qa in tqdm(select_questions(conv, args.categories, args.questions),
                       desc=f"  QA {sample_id}", leave=False):
            t_q = time.time()
            retrieved, context_chars = [], 0
            try:
                out = answer(memory, llm, qa["question"], sample_id, args.top_k)
                pred, retrieved, context_chars = out["answer"], out["retrieved"], out["context_chars"]
            except Exception as e:
                pred = f"(error: {e})"
            record = make_record(sample_id, qa, pred, retrieved=retrieved,
                                 context_chars=context_chars,
                                 seconds=round(time.time() - t_q, 2))
            records.append(record)
            append_trace(trace_path, record)

        save_results(run_dir, {
            "system": "mem0 (single-pass)", "model": args.model, "top_k": args.top_k,
            "summary": summarize(records), "ingestion": ingest_summary, "records": records,
        })

    print_report("mem0 (single-pass)", records)


if __name__ == "__main__":
    main()
