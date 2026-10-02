#!/usr/bin/env python3
"""Run the upstream robust A-Mem memory system on LoCoMo with Gemini on Vertex."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import pickle
import re
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
DATASET = UPSTREAM / "data" / "locomo10.json"
RUNS = ROOT / ".runs"
MODELS = ("gemini-3.5-flash", "gemini-3.8-flash")

# Upstream's SentenceTransformer downloads and all run products stay in this folder.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
sys.path.insert(0, str(UPSTREAM))

from load_dataset import parse_conversation  # noqa: E402
from scoring import score_answer  # noqa: E402
from llm_text_parsers import parse_keywords_response, parse_plain_text_answer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_records(path: Path, records: list[dict]) -> None:
    """Publish a deterministic merged trace without exposing a partial file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_records(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            key = (record["sample_id"], record["qa_index"])
            if key in records:
                raise ValueError(f"Duplicate question in {path}:{line_number}: {key}")
            records[key] = record
    return records


def _config(args: argparse.Namespace, model: str) -> dict:
    source_files = ("memory_layer.py", "memory_layer_robust.py", "llm_text_parsers.py")
    return {
        "model": model,
        "backend": "gemini_vertex",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieve_k": args.retrieve_k,
        "conversations": args.conversations,
        "questions_per_conversation": args.questions,
        "categories": args.categories,
        "work_unit": "conversation",
        "dataset_sha256": _sha256(DATASET),
        "upstream_commit": (UPSTREAM / "COMMIT").read_text().strip(),
        "source_sha256": {name: _sha256(UPSTREAM / name) for name in source_files},
        "runner_sha256": _sha256(Path(__file__)),
        "scoring_sha256": _sha256(ROOT / "scoring.py"),
        "generation": {"max_output_tokens": 32000, "temperature": "omitted"},
    }


def _sessions(sample: dict) -> list:
    conversation = parse_conversation(sample["conversation"])
    return [conversation.sessions[key] for key in sorted(conversation.sessions)]


def _questions(sample: dict, args: argparse.Namespace) -> list:
    selected = [(index, question) for index, question in enumerate(sample["qa"])
                if question["category"] in args.categories]
    return selected[:args.questions] if args.questions else selected


def _save_memory(system, cache_dir: Path, next_session: int, previous: dict | None) -> None:
    """Commit a session checkpoint; progress is written last to survive interruption."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"session_{next_session:03d}"
    memory_file = cache_dir / f"{stem}.memories.pkl"
    retriever_file = cache_dir / f"{stem}.retriever.pkl"
    embeddings_file = cache_dir / f"{stem}.embeddings.npy"
    with memory_file.open("wb") as stream:
        pickle.dump(system.memories, stream, protocol=pickle.HIGHEST_PROTOCOL)
    system.retriever.save(str(retriever_file), str(embeddings_file))
    progress = {"next_session": next_session, "memory_file": memory_file.name,
                "retriever_file": retriever_file.name, "embeddings_file": embeddings_file.name}
    _write_json(cache_dir / "progress.json", progress)
    if previous:
        for field in ("memory_file", "retriever_file", "embeddings_file"):
            (cache_dir / previous[field]).unlink(missing_ok=True)


def _load_or_build_memory(model: str, sample: dict, cache_dir: Path):
    from memory_layer_robust import RobustAgenticMemorySystem

    system = RobustAgenticMemorySystem(
        model_name="all-MiniLM-L6-v2", llm_backend="gemini_vertex", llm_model=model,
    )
    sessions = _sessions(sample)
    progress_file = cache_dir / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else None
    start = progress["next_session"] if progress else 0
    if start > len(sessions):
        raise ValueError(f"Invalid checkpoint in {progress_file}")
    if progress:
        with (cache_dir / progress["memory_file"]).open("rb") as stream:
            system.memories = pickle.load(stream)
        system.retriever.load(str(cache_dir / progress["retriever_file"]),
                              str(cache_dir / progress["embeddings_file"]))
        if (system.retriever.embeddings is None or
                len(system.memories) != len(system.retriever.corpus) or
                len(system.memories) != len(system.retriever.embeddings)):
            raise ValueError(f"Memory and retriever cache disagree in {cache_dir}")

    for index in range(start, len(sessions)):
        session = sessions[index]
        print(f"  [{model}] ingest {sample['sample_id']} session {index + 1}/{len(sessions)}", flush=True)
        for turn in session.turns:
            # Preserve the official robust runner's memory note text and timestamps.
            content = "Speaker " + turn.speaker + "says : " + turn.text
            system.add_note(content, time=session.date_time)
        _save_memory(system, cache_dir, index + 1, progress)
        progress = json.loads(progress_file.read_text())
    return system


def _answer(system, question: str, category: int, retrieve_k: int) -> tuple[str, str, str]:
    """Upstream keyword retrieval and category prompts, without category 5 answer leakage."""
    query_prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
    keywords = parse_keywords_response(system.llm_controller.llm.get_completion(query_prompt))
    context = system.find_related_memories_raw(keywords, k=retrieve_k)
    if category == 5:
        prompt = (f"Based on the context: {context}, answer the following question. {question}\n\n"
                  "If the context does not establish the answer, say: I don't know. Short answer:")
    elif category == 2:
        prompt = (f"Based on the context: {context}, answer the following question. "
                  "Use DATE of CONVERSATION to answer with an approximate date.\n"
                  "Please generate the shortest possible answer, using words from the "
                  "conversation where possible, and avoid using any subjects.\n\n"
                  f"Question: {question} Short answer:")
    elif category == 3:
        prompt = (f"Based on the context: {context}, write an answer in the form of a "
                  "short phrase for the following question. Answer with exact words from "
                  f"the context whenever possible.\n\nQuestion: {question} Short answer:")
    else:
        prompt = (f"Based on the context: {context}, write an answer in the form of a "
                  "short phrase for the following question. Answer with exact words from "
                  f"the context whenever possible.\n\nQuestion: {question} Short answer:")
    response = system.llm_controller.llm.get_completion(prompt,
                                                        temperature=0.5 if category == 5 else 0.7)
    return parse_plain_text_answer(response), keywords, context


def _summary(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record["score"])
    return {
        "n": len(records),
        "overall": sum(record["score"] for record in records) / len(records) if records else 0.0,
        "by_category": {str(category): {"n": len(scores), "score": sum(scores) / len(scores)}
                        for category, scores in sorted(groups.items())},
    }


@contextmanager
def _run_lock(run_dir: Path):
    """Keep separate invocations from writing the same model's checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{run_dir} is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _prepare_run(args: argparse.Namespace, model: str) -> Path:
    run_dir = RUNS / args.run / model
    config_path = run_dir / "config.json"
    config = _config(args, model)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"{run_dir} has a different configuration; choose another --run name")
    else:
        _write_json(config_path, config)
    (run_dir / "traces").mkdir(exist_ok=True)
    return run_dir


def _run_sample(args: argparse.Namespace, model: str, sample: dict, run_dir_name: str) -> str:
    """Own one conversation's ordered ingestion, questions, and durable trace."""
    run_dir = Path(run_dir_name)
    sample_id = sample["sample_id"]
    trace_path = run_dir / "traces" / f"{sample_id}.jsonl"
    records = _read_records(trace_path)
    questions = _questions(sample, args)
    expected = {(sample_id, index) for index, _ in questions}
    if not set(records) <= expected:
        raise ValueError(f"Unexpected questions in {trace_path}")
    if len(records) == len(questions):
        return sample_id
    print(f"[{model}] {sample_id} ({len(questions)} questions)", flush=True)
    system = _load_or_build_memory(model, sample, run_dir / "memory" / sample_id)
    for index, qa in questions:
        key = (sample_id, index)
        if key in records:
            continue
        started = time.monotonic()
        prediction, keywords, context = _answer(system, qa["question"],
                                                qa["category"], args.retrieve_k)
        reference = qa.get("adversarial_answer") if qa["category"] == 5 else qa.get("answer")
        record = {
            "sample_id": sample_id, "qa_index": index,
            "question": qa["question"], "reference": reference,
            "response": prediction, "category": qa["category"],
            "score": score_answer(prediction, reference or "", qa["category"]),
            "keywords": keywords, "retrieved_context": context,
            "seconds": round(time.monotonic() - started, 2),
        }
        _append_record(trace_path, record)
        records[key] = record
        if len(records) % 25 == 0:
            print(f"  [{model}] {sample_id}: answered {len(records)} questions", flush=True)
    return sample_id


def _merge_model(args: argparse.Namespace, model: str, samples: list[dict], run_dir: Path) -> None:
    """Publish one result in source order only after all conversations finish."""
    ordered = []
    for sample in samples:
        sample_id = sample["sample_id"]
        trace_path = run_dir / "traces" / f"{sample_id}.jsonl"
        records = _read_records(trace_path)
        questions = _questions(sample, args)
        expected = {(sample_id, index) for index, _ in questions}
        if set(records) != expected:
            raise ValueError(f"Incomplete or unexpected questions in {trace_path}")
        ordered.extend(records[(sample_id, index)] for index, _ in questions)
    _write_records(run_dir / "trace.jsonl", ordered)
    summary = _summary(ordered)
    _write_json(run_dir / "summary.json", {"model": model, "run": args.run,
                                           "scoring": "LoCoMo paper F1/abstention",
                                           "summary": summary})
    print(f"[{model}] {summary['n']} questions, overall {summary['overall']:.4f} -> "
          f"{run_dir / 'summary.json'}", flush=True)


def _run_parallel(args: argparse.Namespace, samples: list[dict], worker=_run_sample) -> None:
    """Bound each model to independent conversation workers."""
    with ExitStack() as locks:
        run_dirs = {}
        for model in args.models:
            run_dir = RUNS / args.run / model
            locks.enter_context(_run_lock(run_dir))
            run_dirs[model] = _prepare_run(args, model)

        pending = {model: deque(samples) for model in args.models}
        failures = []
        active = {}
        capacity = len(args.models) * min(args.workers_per_model, len(samples))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=capacity, mp_context=context) as pool:
            def submit_next(model: str) -> None:
                if pending[model]:
                    sample = pending[model].popleft()
                    future = pool.submit(worker, args, model, sample, str(run_dirs[model]))
                    active[future] = (model, sample["sample_id"])

            for model in args.models:
                for _ in range(min(args.workers_per_model, len(samples))):
                    submit_next(model)
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                succeeded = []
                for future in done:
                    model, sample_id = active.pop(future)
                    try:
                        completed_id = future.result()
                        if completed_id != sample_id:
                            raise ValueError(f"Worker returned {completed_id!r} for {sample_id!r}")
                    except Exception as error:
                        failures.append((model, sample_id, error))
                        print(f"[{model}] {sample_id} failed: {error}", file=sys.stderr, flush=True)
                    else:
                        succeeded.append(model)
                for model in succeeded:
                    if not any(failed_model == model for failed_model, _, _ in failures):
                        submit_next(model)

        for model in args.models:
            if not any(failed_model == model for failed_model, _, _ in failures):
                _merge_model(args, model, samples, run_dirs[model])
        if failures:
            details = "; ".join(f"{model}/{sample_id}: {error}" for model, sample_id, error in failures)
            raise RuntimeError(f"A-Mem workers failed; rerun the same command to resume: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--conversations", type=int, default=10)
    parser.add_argument("--questions", type=int, help="First N questions per conversation")
    parser.add_argument("--categories", type=int, nargs="+", choices=range(1, 6),
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--workers-per-model", type=int, default=2,
                        help="Independent conversation workers for each model (default: 2)")
    parser.add_argument("--run", default="full", help="Run name within evaluations/a_mem/.runs")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="Validate setup without paid API calls")
    checks.add_argument("--check-api", action="store_true", help="Test access to both models, then exit")
    args = parser.parse_args()
    if not 1 <= args.conversations <= 10 or (args.questions is not None and args.questions < 1):
        parser.error("--conversations must be 1..10 and --questions must be positive")
    if args.retrieve_k < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("--retrieve-k must be positive and --run must be a simple name")
    if args.workers_per_model < 1:
        parser.error("--workers-per-model must be positive")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    samples = json.loads(DATASET.read_text(encoding="utf-8"))[:args.conversations]
    if args.check:
        from memory_layer_robust import RobustLLMController
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        RobustLLMController(backend="gemini_vertex", model=args.models[0])
        print(f"Local setup OK: {len(samples)} LoCoMo conversations; models: {', '.join(args.models)}. "
              "API credentials and model access were not checked.")
        return 0
    from memory_layer_robust import RobustLLMController
    for model in args.models:
        print(f"Checking Vertex access to {model}", flush=True)
        RobustLLMController(backend="gemini_vertex", model=model, check_connection=True)
    if args.check_api:
        print("Vertex access OK for all selected models.")
        return 0
    _run_parallel(args, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
