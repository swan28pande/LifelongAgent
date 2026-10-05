"""Grade answers from TiMem's own LoCoMo pipeline with the shared judge and LoCoMo F1.

    python evaluations/timem/grade.py --answers baselines/TiMem/logs/timem_locomo_answers.jsonl \
        --model gemini-3.5-flash --run pilot_conv26

Answers come from baselines/TiMem/experiments/datasets/locomo/02_memory_retrieval.py, which
writes one JSON line per answered question. Questions are matched to locomo10.json by
conversation id and question text. Grading resumes from grades.jsonl in the output folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluations.judge_existing.run import (  # noqa: E402
    CATEGORY_NAMES, INCLUDED_CATEGORIES, Task, dataset_questions, grade_pending,
    previous_grades, save_json, sha256, summarize,
)
from experiments.core.grading import locomo_f1  # noqa: E402
from experiments.core.types import Question  # noqa: E402


def load_tasks(answers: Path) -> tuple[list[Task], int, str]:
    expected, dataset_hash = dataset_questions()
    # LoCoMo repeats some question texts within a conversation (identical answers); copies are
    # matched to dataset positions in the order TiMem answered them.
    by_text: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for (sample_id, index), qa in sorted(expected.items()):
        by_text.setdefault((sample_id, qa["question"]), []).append((index, qa))
    records = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: (r["test_info"]["conversation_id"], r["test_info"]["question_index"]))
    tasks, seen, unmatched = [], set(), 0
    for record in records:
        info = record["test_info"]
        candidates = by_text.get((info["conversation_id"], info["question"]))
        if not candidates:
            unmatched += 1
            continue
        index, qa = candidates.pop(0)
        key = (info["conversation_id"], index)
        if key in seen:
            raise ValueError(f"Duplicate answer for {key} in {answers}")
        seen.add(key)
        if qa["category"] not in INCLUDED_CATEGORIES:
            continue
        answer = str(qa["answer"])
        q = Question(id=f"{key[0]}_q{index}", question=qa["question"], answer=answer, accept=[answer],
                     answer_type="free_text", grading="locomo_f1",
                     category=CATEGORY_NAMES[qa["category"]], meta={"category_id": qa["category"]})
        response = record["response"]["answer"] or ""
        tasks.append(Task(q, key[0], qa["category"], response, float(locomo_f1(q, response) or 0.0)))
    tasks.sort(key=lambda t: t.question.id)
    return tasks, unmatched, dataset_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    tasks, unmatched, dataset_hash = load_tasks(args.answers)
    if unmatched:
        raise ValueError(f"{unmatched} answers did not match a LoCoMo question")
    out = ROOT / "evaluations" / "timem" / "runs" / args.model / args.run
    out.mkdir(parents=True, exist_ok=True)
    grades = previous_grades(out / "grades.jsonl", tasks)
    asyncio.run(grade_pending(tasks, grades, out, args.workers))
    metadata = {"source": str(args.answers), "source_sha256": sha256(args.answers.read_bytes()),
                "system": "timem", "answer_model": args.model, "dataset_sha256": dataset_hash,
                "question_count": len(tasks), "f1": "experiments.core.grading.locomo_f1"}
    summary = summarize(tasks, grades, metadata)
    save_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("question_count", "overall", "by_category")}, indent=2))


if __name__ == "__main__":
    main()
