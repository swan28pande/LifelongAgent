"""Monthly barriers shared by memory_v4 and the three baseline methods."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

from experiments import config
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt, strict_check
from experiments.core.types import Instance

from .loader import Timeline
from .methods import METHODS, create


SETUP_DIR = Path(__file__).resolve().parent
PROTOCOL = 'monthly_cumulative_original_probe_date'


def _method(store, instance, model, usage):
    from experiments.methods.ours_v4d import OursV4D
    return OursV4D(store, instance, model, usage)


def _judge(model, log):
    from generator_v2.llm import LLM
    return LLM(model, 0.0, log)


def _usage():
    from experiments.core.method import UsageCounter
    return UsageCounter()


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _append(path: Path, value) -> None:
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _log(path: Path) -> dict[str, dict]:
    """Recover an incomplete last append; reject corruption in committed lines."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        tail = data[data.rfind(b'\n') + 1:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            data = data[:data.rfind(b'\n') + 1]
            with path.open('r+b') as f:
                f.truncate(len(data))
        else:
            with path.open('ab') as f:
                f.write(b'\n')
    records = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record['id'] in records:
            raise RuntimeError(f'Duplicate record {record["id"]} in {path}')
        records[record['id']] = record
    return records


@contextmanager
def _lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Another process is using {output}') from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def _measure(directory: Path, stage: str, usage):
    before = usage.snapshot()
    started = time.monotonic()
    try:
        yield
    finally:
        after = usage.snapshot()
        metrics = _read(directory / 'usage.json', {})
        metrics[stage + '_seconds'] = metrics.get(stage + '_seconds', 0.0) + time.monotonic() - started
        for key, value in after.items():
            metrics[key] = metrics.get(key, 0) + value - before.get(key, 0)
        _write(directory / 'usage.json', metrics)


def _backup(store: Path, backup: Path) -> None:
    if backup.exists():
        shutil.rmtree(backup)
    if store.exists():
        shutil.copytree(store, backup)
    else:
        backup.mkdir()


def _restore(store: Path, backup: Path) -> None:
    if not backup.is_dir():
        raise RuntimeError('Missing checkpoint backup; use a new run directory')
    if store.exists():
        shutil.rmtree(store)
    shutil.copytree(backup, store)


def _validate_results(checkpoint, directory: Path, complete: bool = False):
    answers, grades = _log(directory / 'answers.jsonl'), _log(directory / 'grades.jsonl')
    questions = {q.id: q for q in checkpoint.questions}
    if set(answers) - questions.keys() or set(grades) - answers.keys():
        raise RuntimeError(f'Unexpected answer/grade IDs in {directory}')
    for qid, answer in answers.items():
        q = questions[qid]
        if (answer['checkpoint'] != checkpoint.month or answer['cutoff'] != checkpoint.cutoff
                or answer['asked_on'] != q.asked_on or answer['question'] != q.question
                or answer['reference'] != q.answer or answer['accept'] != q.accept):
            raise RuntimeError(f'Answer metadata changed for {qid} in {directory}')
    for grade in grades.values():
        if grade['checkpoint'] != checkpoint.month:
            raise RuntimeError(f'Wrong checkpoint in {directory}')
    if complete and (set(answers) != questions.keys() or set(grades) != questions.keys()):
        raise RuntimeError(f'Completed checkpoint is missing results: {directory}; use a new run')
    return answers, grades


def _answer(method, checkpoint, directory: Path, workers: int) -> None:
    answered, _ = _validate_results(checkpoint, directory)
    todo = [q for q in checkpoint.questions if q.id not in answered]
    if not todo:
        return

    def one(q):
        started = time.monotonic()
        result = method.answer(q)  # Existing adapter uses the ORIGINAL asked_on.
        meta = dict(result.meta or {})
        if 'tool_calls' in meta:
            meta['num_tool_calls'] = len(meta['tool_calls'])
        return {
            'id': q.id, 'checkpoint': checkpoint.month, 'cutoff': checkpoint.cutoff,
            'asked_on': q.asked_on, 'question': q.question, 'reference': q.answer,
            'accept': q.accept, 'category': q.category, 'response': result.text,
            'meta': meta, 'seconds': time.monotonic() - started,
        }

    failure = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, q): q.id for q in todo}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                record = future.result()
            except Exception as e:
                failure = failure or e
                for pending in futures:
                    pending.cancel()
            else:
                _append(directory / 'answers.jsonl', record)
    if failure:
        raise RuntimeError(f'Answer failed; rerun to resume: {failure}') from failure


async def _grade(judge, checkpoint, directory: Path, concurrency: int) -> None:
    answers, grades = _validate_results(checkpoint, directory)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(q):
        if q.id in grades:
            return
        response = answers[q.id]['response']
        started = time.monotonic()
        async with semaphore:
            verdict, usage = await judge.json(
                JUDGE_SYSTEM, judge_prompt(q, response), f'judge {checkpoint.month}/{q.id}'
            )
        if verdict.get('verdict') not in ('RIGHT', 'WRONG'):
            raise ValueError(f'Invalid judge verdict for {q.id}: {verdict}')
        _append(directory / 'grades.jsonl', {
            'id': q.id, 'checkpoint': checkpoint.month,
            'judge': verdict['verdict'] == 'RIGHT', 'why': verdict.get('why', ''),
            'strict': strict_check(q, response), 'seconds': time.monotonic() - started,
            'usage': {key: getattr(usage, key, 0) for key in
                      ('input_tokens', 'output_tokens', 'reasoning_tokens')},
        })

    results = await asyncio.gather(*(one(q) for q in checkpoint.questions), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise RuntimeError(f'Judge failed; rerun to resume: {failures[0]}') from failures[0]


def _scores(rows) -> dict:
    if not rows:
        return {'n': 0, 'accuracy': None}
    return {'n': len(rows), 'accuracy': sum(r['judge'] for r in rows) / len(rows)}


def _summarize(checkpoint, directory: Path, method_name: str = 'ours_v4d') -> dict:
    answers, grades = _validate_results(checkpoint, directory, complete=True)
    by_category = {}
    for q in checkpoint.questions:
        by_category.setdefault(q.category, []).append(grades[q.id])
    tools = [c for a in answers.values() for c in a['meta'].get('tool_calls', [])]
    by_tool = {}
    for call in tools:
        by_tool[call['tool']] = by_tool.get(call['tool'], 0) + 1
    latencies = sorted(a['seconds'] for a in answers.values())
    metrics = _read(directory / 'usage.json', {})
    judge_log = directory / 'judge_log.jsonl'
    if judge_log.exists():
        calls = [json.loads(line) for line in judge_log.read_text().splitlines() if line.strip()]
        judge_usage = {'calls': len(calls), **{
            key: sum(r.get(key, 0) for r in calls)
            for key in ('input_tokens', 'output_tokens', 'reasoning_tokens')
        }}
    else:
        judge_usage = {'calls': len(grades), **{
            key: sum(r['usage'].get(key, 0) for r in grades.values())
            for key in ('input_tokens', 'output_tokens', 'reasoning_tokens')
        }}
    return {
        'method': method_name, 'method_profile': METHODS[method_name][2],
        'month': checkpoint.month, 'cutoff': checkpoint.cutoff,
        'new_sessions': len(checkpoint.sessions), 'new_questions': len(checkpoint.new_question_ids),
        **_scores(list(grades.values())),
        'new': _scores([grades[qid] for qid in checkpoint.new_question_ids]),
        'repeated': _scores([g for qid, g in grades.items() if qid not in checkpoint.new_question_ids]),
        'by_category': {key: _scores(value) for key, value in sorted(by_category.items())},
        'tool_calls': {'total': len(tools), 'by_tool': by_tool},
        'latency': ({'mean': sum(latencies) / len(latencies), 'p50': latencies[len(latencies) // 2],
                     'p95': latencies[min(int(len(latencies) * .95), len(latencies) - 1)]}
                    if latencies else None),
        'usage': metrics, 'judge_usage': judge_usage,
    }


def _overall(timelines, output: Path, through, method_name: str = 'ours_v4d') -> dict:
    users, by_month = {}, {}
    for timeline in timelines:
        state = _read(output / timeline.id / 'state.json', {'completed': 0})
        months = []
        for checkpoint in timeline.checkpoints[:state['completed']]:
            if through is not None and checkpoint.month > through:
                continue
            row = _read(output / timeline.id / 'months' / checkpoint.month / 'summary.json', None)
            if row is None:
                raise RuntimeError(f'Missing summary for completed {timeline.id}/{checkpoint.month}')
            months.append(row)
            by_month.setdefault(checkpoint.month, []).append(row)
        users[timeline.id] = months
    all_rows = [row for rows in users.values() for row in rows]

    def combined(rows):
        n = sum(r['n'] for r in rows)
        return {'n': n, 'accuracy': sum(r['accuracy'] * r['n'] for r in rows if r['n']) / n if n else None}

    requested = [(t, c) for t in timelines for c in t.checkpoints if through is None or c.month <= through]
    complete = all(any(r['month'] == c.month for r in users[t.id]) for t, c in requested)
    return {
        'protocol': PROTOCOL, 'method': method_name, 'method_profile': METHODS[method_name][2],
        'complete': complete, 'through': through,
        **combined(all_rows), 'by_month': {m: combined(v) for m, v in sorted(by_month.items())},
        'by_user': users,
        'usage': {key: sum(r['usage'].get(key, 0) for r in all_rows)
                  for key in ('calls', 'input_tokens', 'output_tokens')},
        'judge_usage': {key: sum(r['judge_usage'].get(key, 0) for r in all_rows)
                        for key in ('calls', 'input_tokens', 'output_tokens', 'reasoning_tokens')},
        'stage_seconds': {stage: sum(r['usage'].get(stage + '_seconds', 0) for r in all_rows)
                          for stage in ('ingest', 'finalize', 'answer', 'judge')},
    }


def _run_user(timeline, output, model, judge_model, through, workers, judge_concurrency,
              method_name, method_factory, judge_factory, usage_factory):
    method = None
    try:
        user_dir = output / timeline.id
        user_dir.mkdir(exist_ok=True)
        store, backup = user_dir / 'store', user_dir / 'checkpoint_backup'
        state_path = user_dir / 'state.json'
        if not state_path.exists() and any(user_dir.iterdir()):
            raise RuntimeError(f'Missing checkpoint state for {timeline.id}; use a new run')
        state = _read(state_path, {'completed': 0, 'phase': 'idle'})
        completed = state['completed']
        if (not isinstance(completed, int) or not 0 <= completed <= len(timeline.checkpoints)
                or state['phase'] not in ('idle', 'building', 'evaluating')):
            raise RuntimeError(f'Invalid checkpoint state for {timeline.id}')
        for prior in timeline.checkpoints[:completed]:
            _validate_results(prior, user_dir / 'months' / prior.month, complete=True)
        if completed and not store.is_dir() and state['phase'] != 'building':
            raise RuntimeError(f'Missing persistent store for {timeline.id}')
        if state['phase'] != 'idle':
            if completed == len(timeline.checkpoints) or state['month'] != timeline.checkpoints[completed].month:
                raise RuntimeError(f'Invalid active month for {timeline.id}')

        usage = usage_factory()
        method = None
        for index, checkpoint in enumerate(timeline.checkpoints):
            if index < completed or (through is not None and checkpoint.month > through):
                continue
            directory = user_dir / 'months' / checkpoint.month
            directory.mkdir(parents=True, exist_ok=True)
            print(f'[{timeline.id}/{checkpoint.month}] {len(checkpoint.sessions)} new sessions, '
                  f'{len(checkpoint.questions)} cumulative questions', flush=True)
            if state['phase'] == 'idle':
                _backup(store, backup)  # Prepare undo before any agent writes.
                state = {'completed': index, 'phase': 'building', 'month': checkpoint.month}
                _write(state_path, state)
            elif state['phase'] == 'building':
                _restore(store, backup)
                (directory / 'ingest.jsonl').unlink(missing_ok=True)
            if state['month'] != checkpoint.month:
                raise RuntimeError(f'Checkpoint frontier mismatch for {timeline.id}')

            if method is None:
                # Keep one agent per user; construction receives only identity.
                instance = Instance(timeline.id, [], [], timeline.name)
                method = method_factory(store, instance, model, usage)
            if state['phase'] == 'building':
                for session in checkpoint.sessions:
                    with _measure(directory, 'ingest', usage):
                        stats = method.ingest(session)
                    _append(directory / 'ingest.jsonl', {
                        'id': session.id, 'date': session.date, 'stats': stats or {},
                    })
                with _measure(directory, 'finalize', usage):
                    method.finalize()  # Finish this method's memory before answering.
                state = {'completed': index, 'phase': 'evaluating', 'month': checkpoint.month}
                _write(state_path, state)
            if backup.exists():
                shutil.rmtree(backup)

            with _measure(directory, 'answer', usage):
                _answer(method, checkpoint, directory, workers)
            _, grades = _validate_results(checkpoint, directory)
            if len(grades) != len(checkpoint.questions):
                with _measure(directory, 'judge', usage):
                    judge = judge_factory(judge_model, directory / 'judge_log.jsonl')
                    asyncio.run(_grade(judge, checkpoint, directory, judge_concurrency))
            summary = _summarize(checkpoint, directory, method_name)
            _write(directory / 'summary.json', summary)
            state = {'completed': index + 1, 'phase': 'idle'}
            _write(state_path, state)
    finally:
        if method is not None and hasattr(method, 'close'):
            method.close()


def _implementation_hashes(method_name):
    root = SETUP_DIR.parent
    sources = [root / name for name in (
        'experiments/config.py', 'experiments/core/grading.py', 'experiments/core/types.py',
        'experiments/core/method.py', 'generator_v2/llm.py',
    )] + list(SETUP_DIR.glob('*.py'))
    if method_name == 'ours_v4d':
        sources += list((root / 'memory_v4').glob('*.py')) + [
            root / name for name in ('experiments/methods/ours_v4.py',
                                    'experiments/methods/ours_v4d.py', 'experiments/methods/_common.py')
        ]
    else:
        folder = METHODS[method_name][0].split('.')[-2]
        sources += [SETUP_DIR / 'baselines' / name for name in ('_common.py', 'store.py', 'requirements.txt')]
        sources += [p for p in (SETUP_DIR / 'baselines' / folder).rglob('*')
                    if p.suffix in ('.py', '.yaml', '.yml') or p.name == 'requirements.txt']
        # Runtime stores and results never contribute to implementation identity.
        sources = [p for p in sources if not set(p.relative_to(root).parts)
                   & {'results', 'tmp', '.cache', '__pycache__'}]
        if method_name in ('timem', 'naive_rag'):
            sources.append(root / 'memory_v3/store.py')
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(sources))}


def run(timelines: list[Timeline], output: Path, *, model: str = config.MODEL,
        judge_model: str = config.JUDGE_MODEL, workers: int = config.ANSWER_WORKERS,
        judge_concurrency: int = config.JUDGE_CONCURRENCY, through: str | None = None,
        method_name: str = 'ours_v4d', user_workers: int = 1,
        method_factory=None, judge_factory=_judge, usage_factory=_usage) -> dict:
    """Build → answer → judge each month, with independent users optionally parallel."""
    output = Path(output).resolve()
    if not output.is_relative_to(SETUP_DIR) or output == SETUP_DIR:
        raise ValueError('Run output must be a subdirectory of setup_2/')
    if min(workers, judge_concurrency, user_workers) < 1 or not timelines:
        raise ValueError('Select users and positive user/answer/judge concurrency')
    if method_name not in METHODS:
        raise ValueError(f'Unknown method: {method_name}')
    if len({t.id for t in timelines}) != len(timelines):
        raise ValueError('Select each user once')
    if method_factory is None:
        method_factory = _method if method_name == 'ours_v4d' else partial(create, method_name)
    temporary = SETUP_DIR / 'tmp'
    temporary.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('HF_HOME', str(SETUP_DIR / '.cache' / 'huggingface'))
    manifest = {
        'version': 2, 'protocol': PROTOCOL, 'method': method_name,
        'method_profile': METHODS[method_name][2], 'model': model, 'judge_model': judge_model,
        'inputs': {t.id: t.input_hashes for t in timelines},
        'implementation_hashes': _implementation_hashes(method_name),
    }
    with _lock(output):
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            if _read(manifest_path, None) != manifest:
                raise ValueError('Inputs/models/protocol changed (or method/implementation); choose a new run name')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Nonempty output has no manifest; choose a new run name')
            _write(manifest_path, manifest)

        started = time.monotonic()
        failure = None
        jobs = [(t, output, model, judge_model, through, workers, judge_concurrency,
                 method_name, method_factory, judge_factory, usage_factory) for t in timelines]
        try:
            if user_workers == 1 or len(jobs) == 1:
                for job in jobs:
                    _run_user(*job)
            else:
                # Spawn keeps model clients, native TiMem globals and user memory
                # independent. Pool termination also stops children on Ctrl-C.
                with multiprocessing.get_context('spawn').Pool(min(user_workers, len(jobs))) as pool:
                    pool.starmap(_run_user, jobs)
        except BaseException as e:
            failure = {'type': type(e).__name__, 'message': str(e)}
            raise
        finally:
            summary = _overall(timelines, output, through, method_name)
            summary.update({'model': model, 'judge_model': judge_model})
            if failure:
                summary['complete'] = False
                summary['error'] = failure
            summary['seconds_this_invocation'] = time.monotonic() - started
            _write(output / 'summary.json', summary)
        return summary
