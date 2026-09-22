"""
Run memory_v3 over the eval dataset and score the QA pairs.

    python3 scripts/run_memory_v3.py                    # full 60 days, 67 questions
    python3 scripts/run_memory_v3.py --days 10 --limit 15
    python3 scripts/run_memory_v3.py --skip-ingest      # reuse an existing store

Three phases: ingest each day in order, build the summary hierarchy once at the end,
then answer the questions. Scored with token-F1 and an LLM judge, broken down by
question type and difficulty, matching what evaluate_synthetic.py reports for the
other systems so the numbers are comparable.

Everything is written to results/<run>/ as it goes, so a crash in phase 3 does not
cost the ingestion.
"""

import argparse
import json
import os
import shutil
import string
import sys
import time
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from memory_v3.agent import AgenticMemoryAgent, _make_llm

JUDGE_SYSTEM = "You are a strict grading assistant."

JUDGE_PROMPT = """\
Query: {query}
Ground Truth: {truth}
System Prediction: {pred}

Is the System Prediction semantically correct and factually consistent with the
Ground Truth? Ignore minor phrasing differences or verbosity.

EVALUATION GUIDELINES:
- Temporal Evolution: Facts can evolve over time. If the Ground Truth describes a state
  as current/upcoming but the System Prediction describes it as past/completed (or
  vice-versa), treat it as consistent if they refer to the same event.
- State Changes: If the Ground Truth describes a historical preference or state that
  subsequently changed, the prediction is correct if it lists either the relevant state
  or mentions the transition.
- List/Group Overlap: If the Ground Truth is a single item and the System Prediction
  provides a category, list, or set of options that clearly contains the Ground Truth
  item, it is correct.
- Extra Details: Do not penalize the System Prediction for including extra reasoning,
  context, or details, as long as the correct core information is present.

If the System says "I don't know" but the Ground Truth has a correct answer, it is WRONG.
If the System provides the correct core information (even if embedded in a larger
correct context or list), it is RIGHT.

Return only "RIGHT" or "WRONG".
"""


def token_f1(pred: str, truth: str) -> float:
    strip = str.maketrans("", "", string.punctuation)
    p = pred.lower().translate(strip).split()
    t = truth.lower().translate(strip).split()
    common = set(p) & set(t)
    if not common or not p or not t:
        return 0.0
    precision, recall = len(common) / len(p), len(common) / len(t)
    return 2 * precision * recall / (precision + recall)


def llm_judge(llm, query: str, pred: str, truth: str) -> float:
    try:
        prompt = ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM), ("human", JUDGE_PROMPT)]
        )
        result = (prompt | llm).invoke(
            {"query": query, "truth": truth, "pred": pred}
        )
        content = result.content
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        return 1.0 if "RIGHT" in str(content).strip().upper() else 0.0
    except Exception as e:
        print(f"  judge error: {e}")
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="max questions")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--run", default="memory_v3_run")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-summaries", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(PROJECT_ROOT, "results", args.run)
    store_dir = os.path.join(run_dir, "store")
    os.makedirs(run_dir, exist_ok=True)

    if not args.skip_ingest and os.path.exists(store_dir):
        shutil.rmtree(store_dir)

    with open(os.path.join(PROJECT_ROOT, "datasets", "eval", "conversations.json")) as f:
        sessions = json.load(f)["user_1"]["sessions"]
    with open(os.path.join(PROJECT_ROOT, "datasets", "eval", "qa_pairs.json")) as f:
        qa_pairs = json.load(f)["user_1"]["qa_pairs"]

    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    agent = AgenticMemoryAgent(base_dir=store_dir, model=args.model)
    judge_llm = _make_llm(args.model, 0.0)

    # ── Phase 1: ingest ─────────────────────────────────────────────
    if not args.skip_ingest:
        dates = sorted(sessions)[: args.days]
        print(f"--- PHASE 1: INGESTING {len(dates)} DAYS ---")
        t0 = time.time()
        reports = agent.ingest_range(sessions, dates=dates)
        elapsed = time.time() - t0

        stats = {
            "days": len(reports),
            "seconds": round(elapsed, 1),
            "extracted": sum(r.extracted for r in reports),
            "added": sum(r.added for r in reports),
            "duplicates": sum(r.duplicates for r in reports),
            "chunks": sum(r.chunks_indexed for r in reports),
            "errors": [e for r in reports for e in r.errors],
            "entities": agent.store.get_all_entities(),
        }
        print(f"\n  {stats['added']} preferences, {stats['chunks']} chunks, "
              f"{stats['duplicates']} duplicates skipped, {elapsed:.0f}s")
        print(f"  entities: {', '.join(stats['entities'])}")
        if stats["errors"]:
            print(f"  ERRORS: {stats['errors'][:5]}")

        with open(os.path.join(run_dir, "ingest_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(run_dir, "memories.json"), "w") as f:
            json.dump(agent.store.query_memories(limit=5000), f, indent=2)

    # ── Phase 2: summaries ──────────────────────────────────────────
    if not args.skip_summaries:
        print("\n--- PHASE 2: BUILDING SUMMARIES ---")
        t0 = time.time()
        agent.build_summaries(force=True)
        print(f"  done in {time.time() - t0:.0f}s")

    # ── Phase 3: QA ─────────────────────────────────────────────────
    print(f"\n--- PHASE 3: ANSWERING {len(qa_pairs)} QUESTIONS ---")
    records = []
    by_type = defaultdict(list)
    by_difficulty = defaultdict(list)

    for qa in tqdm(qa_pairs, desc="QA"):
        question, truth = qa["question"], qa["answer"]
        try:
            pred = agent.chat(question)
        except Exception as e:
            pred = f"(error: {e})"

        f1 = token_f1(pred, truth)
        judged = llm_judge(judge_llm, question, pred, truth)

        by_type[qa["type"]].append((f1, judged))
        by_difficulty[qa["difficulty"]].append((f1, judged))
        records.append({
            "question": question, "answer": truth, "response": pred,
            "type": qa["type"], "difficulty": qa["difficulty"],
            "f1": f1, "llm": judged,
        })

    # ── Report ──────────────────────────────────────────────────────
    f1s = [r["f1"] for r in records]
    judges = [r["llm"] for r in records]

    def block(title, groups):
        print(f"\n  By {title}:")
        for key in sorted(groups):
            pairs = groups[key]
            print(f"    {key:<24} n={len(pairs):<4} "
                  f"F1 {np.mean([p[0] for p in pairs]):.3f}   "
                  f"judge {np.mean([p[1] for p in pairs]):.3f}")

    print(f"\n{'=' * 64}")
    print(f"  memory_v3 · {args.model}")
    print(f"  Overall (n={len(records)})      "
          f"F1 {np.mean(f1s):.3f}   judge {np.mean(judges):.3f}")
    block("difficulty", by_difficulty)
    block("type", by_type)
    print(f"{'=' * 64}")

    out = {
        "model": args.model,
        "summary": {
            "n": len(records),
            "overall_f1": float(np.mean(f1s)),
            "overall_llm": float(np.mean(judges)),
            "by_type": {k: {"n": len(v),
                            "f1": float(np.mean([p[0] for p in v])),
                            "llm": float(np.mean([p[1] for p in v]))}
                        for k, v in by_type.items()},
            "by_difficulty": {k: {"n": len(v),
                                  "f1": float(np.mean([p[0] for p in v])),
                                  "llm": float(np.mean([p[1] for p in v]))}
                              for k, v in by_difficulty.items()},
        },
        "records": records,
    }
    path = os.path.join(run_dir, "qa_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
