"""Judge saved MemoryOS, Zep, and A-Mem LoCoMo answers without rerunning ingestion.

Uses the same judge model and prompt as experiments/core/runner.py. The shared
experiments LoCoMo protocol evaluates categories 1-4 by default. Use
--include-adversarial to also judge category 5 with the abstention rule and reuse
compatible category 1-4 grades. Source result files are only read.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments import config  # noqa: E402
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt  # noqa: E402
from experiments.core.types import Question  # noqa: E402

SYSTEMS = ("memoryos", "zep", "a_mem")
MODELS = ("gemini-3.5-flash", "gemini-3.8-flash")
INCLUDED_CATEGORIES = (1, 2, 3, 4)
ADVERSARIAL_REFERENCE = "I don't know."
OUTPUT_ROOT = HERE / "runs"
# Match the shared LoCoMo labels without importing its NLTK-based F1 scorer;
# saved-result judging reuses the scores already recorded in each source.
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "commonsense",
                  4: "single-hop", 5: "adversarial"}


@dataclass(frozen=True)
class Task:
    question: Question
    sample_id: str
    category_id: int
    response: str
    source_f1: float


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dataset_questions(include_adversarial: bool = False) -> tuple[dict[tuple[str, int], dict], str]:
    raw = (config.DATA_DIR / "locomo10.json").read_bytes()
    samples = json.loads(raw)
    categories = INCLUDED_CATEGORIES + (5,) if include_adversarial else INCLUDED_CATEGORIES
    questions = {
        (sample["sample_id"], index): qa
        for sample in samples
        for index, qa in enumerate(sample["qa"])
        if qa.get("category") in categories
        and ("adversarial_answer" in qa if qa["category"] == 5 else "answer" in qa)
    }
    count = 1986 if include_adversarial else 1540
    if len(questions) != count:
        raise ValueError(f"Expected {count} LoCoMo questions for categories {categories}, "
                         f"found {len(questions)}")
    return questions, sha256(raw)


def prepare(system: str, model: str, expected: dict, dataset_hash: str,
            include_adversarial: bool = False) -> tuple[list[Task], dict]:
    categories = INCLUDED_CATEGORIES + (5,) if include_adversarial else INCLUDED_CATEGORIES
    if system == "a_mem":
        run_dir = ROOT / "evaluations" / "a_mem" / ".runs" / "full" / model
        source = run_dir / "trace.jsonl"
        raw = source.read_bytes()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        source_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if source_config.get("model") != model or source_config.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"Model or LoCoMo dataset mismatch: {source}")
        if result.get("model") != model or result.get("run") != "full" \
                or result.get("summary", {}).get("n") != len(records):
            raise ValueError(f"Incomplete or mismatched source summary: {source}")
        reference_field = "reference"
    else:
        source = ROOT / "evaluations" / system / "runs" / model / "locomo_results.json"
        raw = source.read_bytes()
        result = json.loads(raw)
        if result.get("system") != {"memoryos": "MemoryOS", "zep": "Zep Cloud"}[system]:
            raise ValueError(f"System mismatch: {source}")
        if result.get("model") != model or result.get("data_sha256") != dataset_hash:
            raise ValueError(f"Model or LoCoMo dataset mismatch: {source}")
        if result.get("summary", {}).get("n") != len(result.get("records", [])):
            raise ValueError(f"Incomplete source summary: {source}")
        records = result["records"]
        reference_field = "answer"

    tasks = []
    seen = set()
    for record in records:
        category = record["category"]
        if category not in categories:
            continue
        key = (record["sample_id"], record["qa_index"])
        if key in seen:
            raise ValueError(f"Duplicate question {key} in {source}")
        seen.add(key)
        qa = expected.get(key)
        reference = str(record[reference_field]) if system == "a_mem" else record[reference_field]
        expected_reference = qa.get("adversarial_answer" if category == 5 else "answer") \
            if qa is not None else None
        if qa is None or category != qa["category"] or record["question"] != qa["question"] \
                or reference != str(expected_reference):
            raise ValueError(f"Question/reference mismatch at {key} in {source}")
        if not isinstance(record.get("response"), str):
            raise ValueError(f"Missing saved response at {key} in {source}")
        # The dataset's adversarial_answer is a distractor, not a supported answer.
        # Validate it against the saved trace, then judge abstention without showing it.
        answer = ADVERSARIAL_REFERENCE if category == 5 else str(qa["answer"])
        question = Question(
            id=f"{key[0]}_q{key[1]}", question=qa["question"],
            answer=answer, accept=[answer],
            answer_type="abstain" if category == 5 else "free_text", grading="locomo_f1",
            category=CATEGORY_NAMES[category], meta={"category_id": category},
        )
        tasks.append(Task(question, key[0], category, record["response"], float(record["score"])))
    if seen != set(expected):
        raise ValueError(f"Source does not cover the full question set for categories {categories}: {source}")
    tasks.sort(key=lambda task: task.question.id)
    metadata = {
        "source": str(source), "source_sha256": sha256(raw),
        "system": system, "answer_model": model,
        "judge_model": config.JUDGE_MODEL, "judge_temperature": 0.0,
        "dataset_sha256": dataset_hash, "categories": list(categories),
        "question_count": len(tasks),
        "judge_system_sha256": sha256(JUDGE_SYSTEM.encode()),
        "judge_prompt_sha256": sha256(inspect.getsource(judge_prompt).encode()),
    }
    if include_adversarial:
        metadata["adversarial_protocol"] = {
            "answer_type": "abstain", "reference": ADVERSARIAL_REFERENCE,
            "accepted": [ADVERSARIAL_REFERENCE],
        }
    return tasks, metadata


def compatible_category_expansion(previous: dict, current: dict) -> bool:
    """Reuse answerable grades only when the entire existing judge protocol matches."""
    if previous.get("categories") != list(INCLUDED_CATEGORIES) \
            or current.get("categories") != [*INCLUDED_CATEGORIES, 5] \
            or previous.get("question_count") != 1540 or current.get("question_count") != 1986:
        return False
    ignored = {"categories", "question_count", "adversarial_protocol"}
    return ({key: value for key, value in previous.items() if key not in ignored}
            == {key: value for key, value in current.items() if key not in ignored})


def previous_grades(path: Path, tasks: list[Task]) -> dict[str, dict]:
    by_id = {task.question.id: task for task in tasks}
    grades = {}
    for grade in load_jsonl(path):
        qid = grade["id"]
        if qid not in by_id or qid in grades:
            raise ValueError(f"Unknown or duplicate saved grade {qid} in {path}")
        if grade.get("response_sha256") != sha256(by_id[qid].response.encode()):
            raise ValueError(f"Saved response changed for {qid} in {path}")
        if grade.get("verdict") not in ("RIGHT", "WRONG"):
            raise ValueError(f"Invalid saved verdict for {qid} in {path}")
        grades[qid] = grade
    return grades


async def grade_pending(tasks: list[Task], grades: dict[str, dict], output: Path,
                        workers: int) -> int:
    pending = [task for task in tasks if task.question.id not in grades]
    if not pending:
        return 0
    from generator_v2.llm import LLM

    judge = LLM(config.JUDGE_MODEL, 0.0, output / "judge_log.jsonl")
    queue: asyncio.Queue[Task] = asyncio.Queue()
    for task in pending:
        queue.put_nowait(task)
    failures = []
    finished = 0

    async def worker() -> None:
        nonlocal finished
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            q = task.question
            for attempt in range(6):
                try:
                    result, _ = await judge.json(JUDGE_SYSTEM, judge_prompt(q, task.response),
                                                 f"judge {q.id}")
                    # Gemini occasionally wraps the requested JSON object in a
                    # one-element array. The verdict and explanation are otherwise
                    # identical; reject arrays with more than one decision.
                    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                        result = result[0]
                    if not isinstance(result, dict):
                        raise ValueError(f"Invalid judge response shape: {type(result).__name__}")
                    verdict = result.get("verdict")
                    if verdict not in ("RIGHT", "WRONG"):
                        raise ValueError(f"Invalid judge verdict: {str(verdict)[:80]}")
                    grade = {
                        "id": q.id, "sample_id": task.sample_id, "category": q.category,
                        "verdict": verdict, "judge": verdict == "RIGHT",
                        "why": str(result.get("why", "")),
                        "source_f1": task.source_f1,
                        "response_sha256": sha256(task.response.encode()),
                    }
                    append_jsonl(output / "grades.jsonl", grade)
                    grades[q.id] = grade
                    finished += 1
                    if finished % 50 == 0 or finished == len(pending):
                        print(f"{output.parent.name}/{output.name}: {len(grades)}/{len(tasks)} judged", flush=True)
                    break
                except Exception as exc:
                    if attempt == 5:
                        failures.append(q.id)
                        print(f"{q.id}: judge failed after 6 attempts: {type(exc).__name__}: {str(exc)[:180]}",
                              flush=True)
                    else:
                        await asyncio.sleep(min(60, 2 ** attempt) + random.uniform(0, 1))
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(workers)))
    if failures:
        raise RuntimeError(f"{len(failures)} judge calls failed; rerun to resume")
    return finished


def summarize(tasks: list[Task], grades: dict[str, dict], metadata: dict) -> dict:
    groups = defaultdict(list)
    for task in tasks:
        groups[task.question.category].append(grades[task.question.id])

    def scores(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "judge_accuracy": round(sum(row["judge"] for row in rows) / len(rows), 4),
            "source_paper_f1": round(sum(row["source_f1"] for row in rows) / len(rows), 4),
        }

    summary = {**metadata, "overall": scores([grades[task.question.id] for task in tasks]),
               "by_category": {name: scores(rows) for name, rows in sorted(groups.items())}}
    if 5 in metadata.get("categories", []):
        summary["answerable"] = scores([grades[task.question.id] for task in tasks
                                        if task.category_id in INCLUDED_CATEGORIES])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--workers", type=int, default=config.JUDGE_CONCURRENCY)
    parser.add_argument("--include-adversarial", action="store_true",
                        help="Include category 5 with abstention grading; reuse compatible saved grades")
    parser.add_argument("--check", action="store_true", help="validate inputs only; make no judge calls")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    expected, dataset_hash = dataset_questions(args.include_adversarial)
    for system in args.systems:
        for model in args.models:
            tasks, metadata = prepare(system, model, expected, dataset_hash, args.include_adversarial)
            output = OUTPUT_ROOT / system / model
            if args.check:
                print(f"OK {system}/{model}: {len(tasks)} aligned saved answers", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            manifest = output / "manifest.json"
            previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
            if previous is not None and previous != metadata \
                    and not compatible_category_expansion(previous, metadata):
                raise ValueError(f"Existing judge run has different inputs or protocol: {manifest}")
            grades = previous_grades(output / "grades.jsonl", tasks)
            if previous != metadata:
                save_json(manifest, metadata)
            print(f"{system}/{model}: {len(grades)} grades reused, {len(tasks) - len(grades)} pending",
                  flush=True)
            asyncio.run(grade_pending(tasks, grades, output, args.workers))
            if len(grades) != len(tasks):
                raise RuntimeError(f"{system}/{model}: incomplete grading")
            summary = summarize(tasks, grades, metadata)
            save_json(output / "summary.json", summary)
            print(f"{system}/{model}: judge accuracy {summary['overall']['judge_accuracy']:.1%} "
                  f"on {len(tasks)} questions", flush=True)


if __name__ == "__main__":
    main()
