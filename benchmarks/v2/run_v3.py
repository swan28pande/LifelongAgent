"""
Run memory_v3 on a generator_v2 user and grade it.

    venv/bin/python benchmarks/v2/run_v3.py --user u1                      # ingest, summarise, answer, grade
    venv/bin/python benchmarks/v2/run_v3.py --user u1 --days 7 --limit 10  # smoke test
    venv/bin/python benchmarks/v2/run_v3.py --user u1 --skip-ingest        # reuse the store, re-answer

Phases:
  1. ingest    every session day in date order (resumable: finished days are recorded)
  2. summaries the weekly → lifetime hierarchy, once at the end
  3. answer    every question, asked after the last day (resumable: one line per answer)
  4. grade     a Gemini Pro judge per question, using the answer type, accepted answers
               and tags from qa_pairs.json; plus a strict string check for exact/date items

Results go to results/v2/<user>/<run>/.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dateutil import parser as dateparser
from tqdm import tqdm

from generator_v2.llm import LLM
from memory_v3.agent import AgenticMemoryAgent

JUDGE_SYSTEM = """\
You grade answers from a personal assistant with long-term memory against a reference.
Judge only whether the system's answer is correct; ignore style and verbosity.

Rules by answer type:
- value: RIGHT if the system's final answer is the reference value (paraphrases count).
  WRONG if it names a different value, or hedges between several values.
- date: RIGHT if the system gives one of the ACCEPTED dates, in any format. WRONG otherwise.
- yes_no: the yes/no must match the reference. WRONG if the explanation contradicts it.
- free_text: RIGHT if it captures the key content of the reference.
  * causes: the same cause, in any words.
  * routines/patterns: the same structure (which values, how many days each, or which
    weekdays). An anchor date is not required.
  * durations: within about 10% of the reference number of days.
  * histories and lists: every item of the reference, no wrong items.
- abstain: RIGHT only if the system says it does not know, it was never mentioned, or
  (for "why" questions) that no reason was given. WRONG if it states or invents an answer.
If the reference has an answer and the system says it does not know, it is WRONG.

Return only JSON: {"verdict": "RIGHT" | "WRONG", "why": "<one short sentence>"}"""

MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
                     rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\s+\d{{4}}\b", re.I)


def dates_in(text: str) -> set[str]:
    out = set()
    for m in DATE_RE.finditer(text):
        try:
            out.add(dateparser.parse(m.group(0), fuzzy=True).date().isoformat())
        except (ValueError, OverflowError):
            pass
    return out


def strict_check(q: dict, pred: str) -> bool | None:
    """String check for exact/date items; None for everything the judge alone grades."""
    if q["grading"] == "date_exact":
        return bool(dates_in(pred) & set(q["accept"]))
    if q["grading"] == "exact":
        p = pred.lower()
        return any(re.search(r"\b" + re.escape(a.lower()) + r"\b", p) for a in q["accept"])
    return None


def judge_prompt(q: dict, pred: str) -> str:
    return (f"QUESTION: {q['question']}\nANSWER TYPE: {q['answer_type']}\n"
            f"REFERENCE: {q['answer']}\nACCEPTED: {json.dumps(q['accept'])}\n"
            f"SYSTEM ANSWER: {pred}")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="u1")
    ap.add_argument("--run", default="v3")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    ap.add_argument("--days", type=int, default=None, help="only the first N session days")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--workers", type=int, default=8, help="parallel questions in phase 3")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-summaries", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "datasets", "v2", args.user)
    run_dir = os.path.join(PROJECT_ROOT, "results", "v2", args.user, args.run)
    store_dir = os.path.join(run_dir, "store")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(data_dir, "conversations.json")) as f:
        sessions = json.load(f)["sessions"]
    with open(os.path.join(data_dir, "qa_pairs.json")) as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]
    dates = sorted(sessions)[: args.days] if args.days else sorted(sessions)

    agent = AgenticMemoryAgent(base_dir=store_dir, model=args.model)

    # ── Phase 1: ingest (resumable) ─────────────────────────────────
    ingest_log = os.path.join(run_dir, "ingest_log.jsonl")
    if not args.skip_ingest:
        done = {r["date"] for r in load_jsonl(ingest_log)}
        todo = [d for d in dates if d not in done]
        print(f"--- PHASE 1: INGEST {len(todo)} of {len(dates)} days ---", flush=True)
        for date in tqdm(todo, desc="ingest"):
            s = sessions[date]
            t0 = time.time()
            report = agent.ingest(date, [{"time_of_day": "All Day",
                                          "turns": [{"speaker": t["speaker"], "text": t["text"]}
                                                    for t in s["turns"]]}])
            if report.errors:
                sys.exit(f"ingest failed on {date}: {report.errors[0][:300]}\n"
                         "Nothing was recorded for this day; fix the error and rerun to resume.")
            append_jsonl(ingest_log, {"date": date, "seconds": round(time.time() - t0, 1),
                                      "extracted": report.extracted, "added": report.added,
                                      "type_counts": report.type_counts, "errors": report.errors})
        with open(os.path.join(run_dir, "memories.json"), "w") as f:
            json.dump(agent.store.query_memories(limit=20000), f, indent=2, ensure_ascii=False)

    # ── Phase 2: summaries ──────────────────────────────────────────
    if not args.skip_summaries:
        print("--- PHASE 2: SUMMARIES ---", flush=True)
        t0 = time.time()
        agent.build_summaries(force=True)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # ── Phase 3: answer (resumable, parallel) ───────────────────────
    answers_path = os.path.join(run_dir, "answers.jsonl")
    answered = {r["id"]: r for r in load_jsonl(answers_path)}
    todo = [q for q in questions if q["id"] not in answered]
    print(f"--- PHASE 3: ANSWER {len(todo)} of {len(questions)} questions ---", flush=True)

    def answer(q: dict) -> dict:
        t0 = time.time()
        try:
            trace = agent.chat_with_trace(q["question"])
            pred, calls = trace["answer"], trace["tool_calls"]
        except Exception as e:
            return {"id": q["id"], "error": f"{type(e).__name__}: {e}"[:300]}
        return {"id": q["id"], "response": pred, "tool_calls": calls, "seconds": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in tqdm(pool.map(answer, todo), total=len(todo), desc="answer"):
            if "error" in rec:
                pool.shutdown(cancel_futures=True)
                sys.exit(f"answering {rec['id']} failed: {rec['error']}\nRerun to resume.")
            append_jsonl(answers_path, rec)
            answered[rec["id"]] = rec

    # ── Phase 4: grade (resumable) ──────────────────────────────────
    grades_path = os.path.join(run_dir, "grades.jsonl")
    graded = {r["id"]: r for r in load_jsonl(grades_path)}
    todo = [q for q in questions if q["id"] not in graded]
    print(f"--- PHASE 4: GRADE {len(todo)} of {len(questions)} answers ---", flush=True)
    judge = LLM(args.judge_model, 0.0, Path(run_dir) / "judge_log.jsonl")

    async def grade_all() -> None:
        sem = asyncio.Semaphore(16)

        async def one(q: dict) -> None:
            pred = answered[q["id"]]["response"]
            async with sem:
                try:
                    out, _ = await judge.json(JUDGE_SYSTEM, judge_prompt(q, pred), f"judge {q['id']}")
                    verdict, why = out.get("verdict") == "RIGHT", out.get("why", "")
                except Exception as e:
                    print(f"  judge failed on {q['id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    return
            rec = {"id": q["id"], "judge": verdict, "why": why, "strict": strict_check(q, pred)}
            append_jsonl(grades_path, rec)
            graded[q["id"]] = rec

        await asyncio.gather(*(one(q) for q in todo))

    asyncio.run(grade_all())
    missing = [q["id"] for q in questions if q["id"] not in graded]
    if missing:
        sys.exit(f"{len(missing)} answers could not be graded; rerun to resume.")

    # ── Report ──────────────────────────────────────────────────────
    rows = []
    for q in questions:
        g, a = graded[q["id"]], answered[q["id"]]
        rows.append({**q, "response": a["response"], "tool_calls": len(a["tool_calls"]),
                     "judge": g["judge"], "strict": g["strict"], "why": g["why"]})

    def table(key) -> dict:
        groups = defaultdict(list)
        for r in rows:
            for k in (r[key] if isinstance(r[key], list) else [r[key]]):
                groups[k].append(r)
        return {k: {"n": len(v), "judge": round(sum(r["judge"] for r in v) / len(v), 3),
                    "strict": (round(sum(bool(r["strict"]) for r in v if r["strict"] is not None)
                                     / max(1, sum(r["strict"] is not None for r in v)), 3)
                               if any(r["strict"] is not None for r in v) else None)}
                for k, v in sorted(groups.items())}

    summary = {
        "user": args.user, "model": args.model, "judge_model": args.judge_model,
        "n": len(rows), "judge_accuracy": round(sum(r["judge"] for r in rows) / len(rows), 3),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 2),
        "by_type": table("type"), "by_capability": table("capability"), "by_tag": table("tags"),
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "records": rows}, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\n  memory_v3 · {args.model} on {args.user}  —  judged by {args.judge_model}")
    print(f"  Overall accuracy {summary['judge_accuracy']:.1%}  (n={len(rows)}, "
          f"{summary['mean_tool_calls']} tool calls per question)")
    for title, key in (("type", "by_type"), ("capability", "by_capability")):
        print(f"\n  By {title}:")
        for k, v in summary[key].items():
            strict = f"   strict {v['strict']:.1%}" if v["strict"] is not None else ""
            print(f"    {k:<24} n={v['n']:<4} judge {v['judge']:.1%}{strict}")
    print(f"{'=' * 70}\nSaved to {run_dir}/results.json")


if __name__ == "__main__":
    main()
