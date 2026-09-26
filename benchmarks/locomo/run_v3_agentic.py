"""
memory_v3 on LoCoMo, agentic read path.

    venv/bin/python benchmarks/locomo/run_v3_agentic.py --conversations 1
    venv/bin/python benchmarks/locomo/run_v3_agentic.py                  # all 10

Each conversation is replayed session by session through the v3 ingestion pipeline,
the summary hierarchy is built, and every question goes to the tool-calling chat agent,
which decides for itself how many retrievals it needs.

Output: results/locomo/<run>/ — locomo_results.json, <sample_id>_trace.jsonl (one line
per answer, with the full tool-call sequence), stores/<sample_id>/.
"""

import argparse
import os
import shutil
import sys
import time
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tqdm import tqdm

from benchmarks.locomo.common import (RESULTS_ROOT, append_trace, get_sessions,
                                      load_dataset, make_record, print_report,
                                      save_results, select_questions,
                                      session_to_turns, summarize)
from memory_v3.agent import AgenticMemoryAgent


def ingest_conversation(agent: AgenticMemoryAgent, conv: dict) -> dict:
    stats = {"sessions": 0, "added": 0, "chunks": 0, "errors": []}
    for date, turns in get_sessions(conv):
        formatted = session_to_turns(turns)
        if not formatted:
            continue
        report = agent.ingest(date, [{"time_of_day": "All Day", "turns": formatted}])
        stats["sessions"] += 1
        stats["added"] += report.added
        stats["chunks"] += report.chunks_indexed
        stats["errors"].extend(report.errors)
    return stats


def tool_stats(records: list) -> dict:
    freq = Counter(tc["tool"] for r in records for tc in r.get("tool_calls", []))
    total = sum(freq.values())
    return {
        "total": total,
        "avg_per_question": round(total / len(records), 2) if records else 0,
        "by_tool": dict(freq),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--questions", type=int, default=None, help="cap questions per conversation")
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--run", default="v3_agentic")
    ap.add_argument("--summaries", action=argparse.BooleanOptionalAction, default=True,
                    help="build the summary hierarchy after ingesting each conversation")
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run)
    os.makedirs(run_dir, exist_ok=True)

    records, ingest_summary = [], []
    for i, conv in enumerate(load_dataset(args.conversations), 1):
        sample_id = conv.get("sample_id", f"conv{i}")
        print(f"\n{'=' * 64}\n[{i}] {sample_id}")

        store_dir = os.path.join(run_dir, "stores", sample_id)
        shutil.rmtree(store_dir, ignore_errors=True)
        os.makedirs(store_dir, exist_ok=True)
        agent = AgenticMemoryAgent(base_dir=store_dir, model=args.model)

        t0 = time.time()
        stats = ingest_conversation(agent, conv)
        stats.update(sample_id=sample_id, seconds=round(time.time() - t0, 1),
                     entities=agent.store.get_all_entities(),
                     speakers=agent.store.get_all_speakers())
        ingest_summary.append(stats)
        print(f"  ingested {stats['sessions']} sessions → {stats['added']} records, "
              f"{stats['chunks']} chunks, {stats['seconds']}s")

        if args.summaries:
            agent.build_summaries(force=True)

        trace_path = os.path.join(run_dir, f"{sample_id}_trace.jsonl")
        open(trace_path, "w").close()
        for qa in tqdm(select_questions(conv, args.categories, args.questions),
                       desc=f"  QA {sample_id}", leave=False):
            t_q = time.time()
            tool_calls = []
            try:
                out = agent.chat_with_trace(qa["question"])
                pred, tool_calls = out["answer"], out["tool_calls"]
            except Exception as e:
                pred = f"(error: {e})"
            record = make_record(sample_id, qa, pred, tool_calls=tool_calls,
                                 num_tool_calls=len(tool_calls),
                                 seconds=round(time.time() - t_q, 2))
            records.append(record)
            append_trace(trace_path, record)

        save_results(run_dir, {
            "system": "memory_v3 (agentic)", "model": args.model,
            "summaries": args.summaries,
            "summary": {**summarize(records), "tool_calls": tool_stats(records)},
            "ingestion": ingest_summary, "records": records,
        })

    print_report("memory_v3 (agentic)", records)
    ts = tool_stats(records)
    print(f"  tool calls: {ts['total']} total, {ts['avg_per_question']} avg/question")


if __name__ == "__main__":
    main()
