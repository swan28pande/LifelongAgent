"""Run one method on one benchmark: ingest → finalize → answer → grade → summarize.

Every stage appends to a JSONL file as it goes and skips work already on disk, so an
interrupted run resumes where it stopped. A failed model call stops the run rather than
being recorded as a result.

Layout: results/experiments/<benchmark>/<method>/<run>/
    stores/<instance>/     the method's memory
    ingest.jsonl           one line per ingested session
    answers.jsonl          one line per answer (with the method's trace)
    grades.jsonl           one line per graded answer
    usage.json             LLM calls and tokens made by the method (not the judge)
    summary.json           accuracy overall, by category, by instance; time and usage
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from generator_v2.llm import LLM

from .. import config
from .grading import JUDGE_SYSTEM, judge_prompt, locomo_f1, strict_check
from .method import UsageCounter
from .types import Instance


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(method_cls, benchmark: str, instances: list[Instance], run_name: str,
        model: str = config.MODEL, judge_model: str = config.JUDGE_MODEL,
        workers: int = config.ANSWER_WORKERS) -> dict:
    out = config.RESULTS_DIR / benchmark / method_cls.name / run_name
    out.mkdir(parents=True, exist_ok=True)
    usage = UsageCounter()
    prior = json.loads((out / "usage.json").read_text()) if (out / "usage.json").exists() else {}

    ingest_log, answers_log, grades_log = out / "ingest.jsonl", out / "answers.jsonl", out / "grades.jsonl"
    answered = {r["id"]: r for r in _load(answers_log)}
    t_start = time.time()

    for inst in instances:
        pending = [q for q in inst.questions if q.id not in answered]
        if not pending:
            continue
        store = out / "stores" / inst.id
        store.mkdir(parents=True, exist_ok=True)
        method = method_cls(store, inst, model, usage)

        done = {r["session"] for r in _load(ingest_log) if r["instance"] == inst.id} if method.persistent else set()
        todo = [s for s in inst.sessions if s.id not in done]
        print(f"[{inst.id}] ingest {len(todo)}/{len(inst.sessions)} sessions", flush=True)
        for s in tqdm(todo, desc=f"ingest {inst.id}", leave=False):
            t0 = time.time()
            try:
                stats = method.ingest(s)
            except Exception as e:
                _save_usage(out, prior, usage)
                sys.exit(f"ingest failed on {inst.id}/{s.id} ({s.date}): {type(e).__name__}: {str(e)[:300]}\n"
                         "Rerun the same command to resume.")
            if method.persistent:
                _append(ingest_log, {"instance": inst.id, "session": s.id, "date": s.date,
                                     "seconds": round(time.time() - t0, 2), **(stats or {})})

        final_marker = store / ".finalized"
        if not method.persistent or not final_marker.exists() or todo:
            print(f"[{inst.id}] finalize", flush=True)
            method.finalize()
            final_marker.touch()

        def answer_one(q):
            t0 = time.time()
            try:
                a = method.answer(q)
            except Exception as e:
                return {"id": q.id, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            return {"id": q.id, "instance": inst.id, "response": a.text, "meta": a.meta,
                    "seconds": round(time.time() - t0, 2)}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rec in tqdm(pool.map(answer_one, pending), total=len(pending), desc=f"answer {inst.id}", leave=False):
                if "error" in rec:
                    pool.shutdown(cancel_futures=True)
                    _save_usage(out, prior, usage)
                    sys.exit(f"answering {rec['id']} failed: {rec['error']}\nRerun to resume.")
                _append(answers_log, rec)
                answered[rec["id"]] = rec
        _save_usage(out, prior, usage)

    questions = {q.id: q for inst in instances for q in inst.questions}
    grades = {r["id"]: r for r in _load(grades_log)}
    todo = [q for q in questions.values() if q.id not in grades and q.id in answered]
    if todo:
        judge = LLM(judge_model, 0.0, out / "judge_log.jsonl")
        asyncio.run(_grade(judge, todo, answered, grades, grades_log))
    missing = [qid for qid in questions if qid not in grades]
    if missing:
        sys.exit(f"{len(missing)} answers are not graded yet; rerun to resume.")

    summary = _summarize(benchmark, method_cls.name, run_name, model, judge_model, instances,
                         answered, grades, time.time() - t_start, prior, usage)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def _grade(judge: LLM, todo, answered, grades, path) -> None:
    sem = asyncio.Semaphore(config.JUDGE_CONCURRENCY)

    async def one(q):
        pred = answered[q.id]["response"]
        async with sem:
            try:
                out, _ = await judge.json(JUDGE_SYSTEM, judge_prompt(q, pred), f"judge {q.id}")
            except Exception as e:
                print(f"  judge failed on {q.id}: {type(e).__name__}", flush=True)
                return
        rec = {"id": q.id, "judge": out.get("verdict") == "RIGHT", "why": out.get("why", ""),
               "strict": strict_check(q, pred), "f1": locomo_f1(q, pred)}
        _append(path, rec)
        grades[q.id] = rec

    await asyncio.gather(*(one(q) for q in todo))


def _save_usage(out: Path, prior: dict, usage: UsageCounter) -> None:
    now = usage.snapshot()
    total = {k: prior.get(k, 0) + now[k] for k in now}
    (out / "usage.json").write_text(json.dumps(total, indent=2))


def _summarize(benchmark, method, run_name, model, judge_model, instances, answered, grades,
               seconds, prior, usage) -> dict:
    rows = [(inst.id, q, grades[q.id]) for inst in instances for q in inst.questions]

    def block(key):
        groups = defaultdict(list)
        for inst_id, q, g in rows:
            groups[key(inst_id, q)].append(g)
        return {k: {"n": len(v), "judge": round(sum(g["judge"] for g in v) / len(v), 4),
                    **({"f1": round(sum(g["f1"] for g in v) / len(v), 4)} if v[0]["f1"] is not None else {})}
                for k, v in sorted(groups.items())}

    judged = [g["judge"] for _, _, g in rows]
    now = usage.snapshot()
    return {
        "benchmark": benchmark, "method": method, "run": run_name,
        "model": model, "judge_model": judge_model,
        "n": len(rows), "accuracy": round(sum(judged) / len(judged), 4),
        "by_category": block(lambda i, q: q.category),
        "by_instance": block(lambda i, q: i),
        "usage": {k: prior.get(k, 0) + now[k] for k in now},
        "seconds_this_run": round(seconds, 1),
    }
