#!/usr/bin/env python3
"""Exercise actual v2 loader and runner offline with memory/judge substitutes.

Substitutes bypass cloud and unavailable LangChain dependencies, not loader/runner
logic. All produced run files are confined to this audit's offline_runner directory.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import types

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

from experiments.benchmarks import v2
from experiments.core.types import Answer, Instance, Question, prompt_for

judge_calls = []


class OfflineLLM:
    def __init__(self, model, temperature, log_path):
        self.model = model

    async def json(self, system, prompt, tag):
        judge_calls.append({"tag": tag, "prompt": prompt})
        # Deliberately demonstrate that summary metric follows judge, even for exact.
        return {"verdict": "WRONG", "why": "Offline stub; no semantic judgment."}, None


fake_llm = types.ModuleType("generator_v2.llm")
fake_llm.LLM = OfflineLLM
sys.modules["generator_v2.llm"] = fake_llm
callbacks = types.ModuleType("langchain_core.callbacks")
callbacks.BaseCallbackHandler = object
sys.modules["langchain_core"] = types.ModuleType("langchain_core")
sys.modules["langchain_core.callbacks"] = callbacks

from experiments.core import runner

ingested, answered = [], []


class OfflineMemory:
    name = "source_audit_offline"
    persistent = False

    def __init__(self, store, inst, model, usage):
        self.store, self.inst = store, inst

    def ingest(self, session):
        ingested.append(session.date)
        return {}

    def finalize(self):
        pass

    def answer(self, q):
        answered.append({"id": q.id, "asked_on": q.asked_on, "seen_session_count": len(ingested),
            "last_seen_date": ingested[-1], "future_sessions_seen": sum(d > q.asked_on for d in ingested),
            "prompt": prompt_for(q)})
        return Answer(q.answer)


def main():
    loaded = v2.load()
    canonical = {f"u{i}": json.loads((ROOT / f"datasets/v2/u{i}/probing_questions.json").read_text()) for i in range(1, 6)}
    monthly_ids = {q["id"] for p in canonical.values() for probe in p["probes"] for q in probe["questions"]}
    loaded_ids = {q.id for i in loaded for q in i.questions}
    metadata_mismatches = []
    for inst in loaded:
        raw = json.loads((ROOT / f"datasets/v2/{inst.id}/qa_pairs.json").read_text())
        originals = {q["id"]: q for q in raw}
        for q in inst.questions:
            r = originals[q.id]
            expected = {"question": r["question"], "answer": r["answer"], "accept": r["accept"],
                "answer_type": r["answer_type"], "grading": r["grading"], "category": r["type"],
                "asked_on": (dt.date.fromisoformat(inst.sessions[0].date) + dt.timedelta(days=r["viewpoint_day"] - 1)).isoformat(),
                "meta": {"capability": r["capability"], "tags": r["tags"], "evidence_days": r["evidence_days"]}}
            for field, value in expected.items():
                if getattr(q, field) != value:
                    metadata_mismatches.append({"id": q.id, "field": field})
    inst = next(i for i in loaded if i.id == "u1")
    selected = canonical["u1"]["probes"][0]["questions"][:2]
    qs = [Question(id=q["id"], question=q["question"], answer=q["answer"], accept=q["accept"],
          answer_type=q["answer_type"], grading=q["grading"], category=q["type"], asked_on=q["probe_date"],
          meta={"capability": q["capability"], "tags": q["tags"], "evidence_days": q["evidence_days"]}) for q in selected]
    runtime = OUT / "offline_runner"
    if runtime.exists():
        assert runtime.is_relative_to(OUT)
        shutil.rmtree(runtime)
    runner.config.RESULTS_DIR = runtime
    summary = runner.run(OfflineMemory, "v2", [Instance(inst.id, inst.sessions, qs, inst.user)],
                         "cutoff_reproduction", model="offline", judge_model="offline", workers=1)
    grades_file = runtime / "v2/source_audit_offline/cutoff_reproduction/grades.jsonl"
    grades = [json.loads(l) for l in grades_file.read_text().splitlines()]
    report = {
        "loader": {"instances": len(loaded), "loaded_questions": len(loaded_ids),
            "canonical_monthly_questions": len(monthly_ids), "monthly_ids_loaded": sorted(loaded_ids & monthly_ids),
            "question_counts_by_user": {i.id: len(i.questions) for i in loaded},
            "asked_on_values": sorted({q.asked_on for i in loaded for q in i.questions}),
            "sessions_by_user": {i.id: len(i.sessions) for i in loaded},
            "chronological_by_user": {i.id: [s.date for s in i.sessions] == sorted(s.date for s in i.sessions) for i in loaded},
            "curated_metadata_mismatches": metadata_mismatches},
        "runner_monthly_cutoff_reproduction": {"tested_ids": [q.id for q in qs], "observations": answered,
            "ingestion_sorted": ingested == sorted(ingested), "grades": grades,
            "judge_calls": judge_calls, "summary_n": summary["n"], "summary_accuracy": summary["accuracy"],
            "limitation": "Monthly questions manually supplied to the actual runner; the normal loader does not supply them. Stub grades test dispatch and storage only."},
        "source_locations": {"loader_qa_pairs": "experiments/benchmarks/v2.py:16", "loader_chronology": "experiments/benchmarks/v2.py:17",
            "loader_metadata": "experiments/benchmarks/v2.py:20", "runner_all_ingest_before_answers": "experiments/core/runner.py:73",
            "runner_answer": "experiments/core/runner.py:99", "grader_judge": "experiments/core/runner.py:150",
            "grader_strict": "experiments/core/grading.py:48", "metric_judge": "experiments/core/runner.py:238"}}
    (OUT / "integration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["loader"], indent=2))
    print(json.dumps(answered, indent=2))
    assert len(loaded_ids) == 250 and not (loaded_ids & monthly_ids)
    assert not metadata_mismatches
    assert all(r["future_sessions_seen"] == 699 for r in answered)
    assert next(g for g in grades if g["id"] == qs[1].id)["strict"] is True
    assert summary["accuracy"] == 0.0


if __name__ == "__main__":
    main()
