"""
memory_v3 on LoCoMo, single-pass read path — the "v3 store, one-pass read" cell.

    venv/bin/python benchmarks/locomo/run_v3_onepass.py --conversations 1

Isolates the store from the agent loop: it reads the stores the agentic run already
built (no re-ingestion, nothing written to them), does one fixed retrieval per question,
and answers with one LLM call using the same prompt as run_mem0.py.

Per question the context is:
  - the top-k conversation chunks by semantic search
  - the top-k summaries by semantic search
  - every preference row (a LoCoMo store holds a few dozen, so all of them fit)

Output: results/locomo/<run>/ — locomo_results.json, <sample_id>_trace.jsonl (one line
per answer, with what was retrieved and the context size).
"""

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from benchmarks.locomo.common import (ANSWER_SYSTEM, RESULTS_ROOT, append_trace,
                                      load_dataset, make_record, print_report,
                                      save_results, select_questions, summarize,
                                      text_of)
from memory_v3.store import MemoryStore
from memory_v3.agent import _make_llm

MAX_PREFERENCE_ROWS = 500


def retrieve(store: MemoryStore, question: str, k_chunks: int, k_summaries: int) -> dict:
    chunks = [d.page_content for d in store.search_conversations(question, k=k_chunks)]
    summaries = [d.page_content for d in store.search_summaries(question, k=k_summaries)]
    prefs = [f"[{r['date']}] ({r['entity']}) {r['content']}"
             for r in store.query_memories(limit=MAX_PREFERENCE_ROWS)]
    return {"chunks": chunks, "summaries": summaries, "preferences": prefs}


def build_context(r: dict) -> str:
    parts = []
    if r["preferences"]:
        parts.append("PREFERENCE RECORDS (dated):\n" + "\n".join(r["preferences"]))
    if r["summaries"]:
        parts.append("SUMMARIES:\n" + "\n\n---\n\n".join(r["summaries"]))
    if r["chunks"]:
        parts.append("CONVERSATION EXCERPTS:\n" + "\n\n---\n\n".join(r["chunks"]))
    return "\n\n".join(parts) or "(nothing retrieved)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--k-chunks", type=int, default=10)
    ap.add_argument("--k-summaries", type=int, default=2)
    ap.add_argument("--store-run", default="v3_agentic",
                    help="run under results/locomo/ whose stores are read")
    ap.add_argument("--run", default="v3_onepass")
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run)
    os.makedirs(run_dir, exist_ok=True)
    llm = _make_llm(args.model, 0.0)
    records = []

    for i, conv in enumerate(load_dataset(args.conversations), 1):
        sample_id = conv.get("sample_id", f"conv{i}")
        store_dir = os.path.join(RESULTS_ROOT, args.store_run, "stores", sample_id)
        if not os.path.exists(os.path.join(store_dir, "memories.db")):
            print(f"[{i}] {sample_id}: no store at {store_dir} — run run_v3_agentic.py first")
            continue
        print(f"\n{'=' * 64}\n[{i}] {sample_id} (store: {args.store_run})")
        store = MemoryStore(store_dir)

        trace_path = os.path.join(run_dir, f"{sample_id}_trace.jsonl")
        open(trace_path, "w").close()
        for qa in tqdm(select_questions(conv, args.categories, args.questions),
                       desc=f"  QA {sample_id}", leave=False):
            t_q = time.time()
            retrieved, context_chars = {}, 0
            try:
                retrieved = retrieve(store, qa["question"], args.k_chunks, args.k_summaries)
                context = build_context(retrieved)
                context_chars = len(context)
                reply = llm.invoke([
                    SystemMessage(content=ANSWER_SYSTEM),
                    HumanMessage(content=f"MEMORY CONTEXT:\n{context}\n\nQUESTION: {qa['question']}"),
                ])
                pred = text_of(reply.content)
            except Exception as e:
                pred = f"(error: {e})"
            record = make_record(sample_id, qa, pred, retrieved=retrieved,
                                 context_chars=context_chars,
                                 seconds=round(time.time() - t_q, 2))
            records.append(record)
            append_trace(trace_path, record)

        save_results(run_dir, {
            "system": "memory_v3 (single-pass)", "model": args.model,
            "store_run": args.store_run, "k_chunks": args.k_chunks,
            "k_summaries": args.k_summaries,
            "summary": summarize(records), "records": records,
        })

    print_report("memory_v3 (single-pass)", records)


if __name__ == "__main__":
    main()
