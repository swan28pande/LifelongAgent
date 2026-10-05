"""Bounded live API comparison; existing baseline and judge code stays unchanged."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
TEMP = ROOT / 'setup_2' / 'tmp' / 'gemini-concurrency-pilot'
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
os.environ.setdefault('HF_HOME', str(ROOT / 'tmp/locomo-v3-update-cache/huggingface'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import google.auth

from experiments import config
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt
from experiments.core.method import UsageCounter
from experiments.core.types import Instance
from generator_v2.llm import LLM
from setup_2.loader import load
from setup_2.methods import create

METHODS = ('timem', 'naive_rag', 'full_context')
REQUEST = ContextVar('pilot_request', default={})


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Recorder:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        self.events = []

    def emit(self, kind, **fields):
        row = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(),
               **REQUEST.get(), **fields}
        with self.lock:
            self.events.append(row)
            with (self.directory / 'events.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')

    def attach(self, llm, method):
        """Observe SDK attempts before its unchanged retry machinery handles them."""
        api = llm.client._api_client
        sync, asynchronous = api._request_once, api._async_request_once

        def response_status(error):
            code = getattr(error, 'code', None)
            return code if isinstance(code, int) else None

        def request(*args, **kwargs):
            start = time.monotonic()
            try:
                result = sync(*args, **kwargs)
            except BaseException as error:
                self.emit('http', method=method, model=llm.model, status=response_status(error),
                          seconds=time.monotonic() - start, error=type(error).__name__)
                raise
            self.emit('http', method=method, model=llm.model, status=200,
                      seconds=time.monotonic() - start)
            return result

        async def async_request(*args, **kwargs):
            start = time.monotonic()
            try:
                result = await asynchronous(*args, **kwargs)
            except BaseException as error:
                self.emit('http', method=method, model=llm.model, status=response_status(error),
                          seconds=time.monotonic() - start, error=type(error).__name__)
                raise
            self.emit('http', method=method, model=llm.model, status=200,
                      seconds=time.monotonic() - start)
            return result

        api._request_once, api._async_request_once = request, async_request

    def call(self, meta, operation):
        token = REQUEST.set(meta)
        start = time.monotonic()
        try:
            result = operation()
        except Exception as error:
            self.emit('logical', ok=False, seconds=time.monotonic() - start,
                      error=f'{type(error).__name__}: {error}'[:500])
            raise
        else:
            self.emit('logical', ok=True, seconds=time.monotonic() - start)
            return result
        finally:
            REQUEST.reset(token)

    async def acall(self, meta, operation):
        token = REQUEST.set(meta)
        start = time.monotonic()
        try:
            result = await operation()
        except Exception as error:
            self.emit('logical', ok=False, seconds=time.monotonic() - start,
                      error=f'{type(error).__name__}: {error}'[:500])
            raise
        else:
            self.emit('logical', ok=True, seconds=time.monotonic() - start)
            return result
        finally:
            REQUEST.reset(token)


class Usage(UsageCounter):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def on_llm_end(self, response, **kwargs):
        super().on_llm_end(response, **kwargs)
        for group in response.generations:
            for generation in group:
                metadata = getattr(getattr(generation, 'message', None), 'usage_metadata', None)
                if metadata is not None:
                    self.recorder.emit('tokens', **metadata)


def summarize(events, selector, seconds):
    selected = [e for e in events if all(e.get(k) == v for k, v in selector.items())]
    logical = [e for e in selected if e['kind'] == 'logical']
    tokens = [e for e in selected if e['kind'] == 'tokens']
    latency = [e['seconds'] for e in logical if e['ok']]
    statuses = Counter(str(e['status']) for e in selected if e['kind'] == 'http')
    inputs = sum(e.get('input_tokens', 0) for e in tokens)
    outputs = sum(e.get('output_tokens', 0) for e in tokens)
    return {
        **selector, 'seconds': seconds, 'requests': len(logical),
        'successful': len(latency), 'failed': sum(not e['ok'] for e in logical),
        'latency_mean_s': statistics.mean(latency) if latency else None,
        'latency_median_s': statistics.median(latency) if latency else None,
        'latency_max_s': max(latency) if latency else None,
        'http_statuses': dict(statuses),
        'input_tokens': inputs, 'output_tokens': outputs,
        'reasoning_tokens': sum(e.get('output_token_details', {}).get('reasoning', 0) for e in tokens),
        'cached_input_tokens': sum(e.get('input_token_details', {}).get('cache_read', 0) for e in tokens),
        'requests_per_minute': 60 * len(latency) / seconds,
        'input_tokens_per_minute': 60 * inputs / seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        parser.error('--run must be a directory name')
    output = Path(__file__).parent / args.run
    output.mkdir(exist_ok=True)
    if (output / 'events.jsonl').exists():
        raise ValueError('Use a fresh run name; measurements must not mix invocations')
    recorder = Recorder(output)
    methods, judges, phases = {}, {}, []
    loop = asyncio.new_event_loop()
    start = time.monotonic()
    timeline = load(users=['u5'])[0]
    checkpoint = timeline.checkpoints[0]
    questions = checkpoint.questions
    instance = Instance(timeline.id, [], [], timeline.name)
    credentials, project = google.auth.default()
    protected = [ROOT / 'experiments/config.py', ROOT / 'experiments/core/grading.py',
                 ROOT / 'generator_v2/llm.py', ROOT / 'setup_2/loader.py',
                 ROOT / 'setup_2/runner.py', ROOT / 'setup_2/baselines/_common.py']
    protected.extend(ROOT / 'setup_2/baselines' / folder / 'method.py'
                     for folder in ('TiMem', 'NaiveRAG', 'DirectPrompting'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    manifest = {
        'project': project, 'quota_project': getattr(credentials, 'quota_project_id', None),
        'location': os.getenv('GOOGLE_CLOUD_LOCATION', 'global'),
        'model': config.MODEL, 'judge_model': config.JUDGE_MODEL,
        'dataset_hashes': timeline.input_hashes, 'user': timeline.id,
        'month': checkpoint.month, 'sessions': len(checkpoint.sessions), 'questions': len(questions),
        'rounds': 2, 'answer_workers_per_method': 8, 'judge_concurrency_per_method': 16,
        'effective_answer_concurrency': len(questions),
        'effective_judge_concurrency': len(questions), 'source_hashes': hashes,
        'execution': 'local; one process with independent baseline stores and model clients',
        'judge_inputs': 'fixed responses from first solitary answer condition',
        'limitation': 'first-month prompts; at most 18 concurrent logical calls, not 120/240',
    }
    save(output / 'manifest.json', manifest)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    def meta(method, stage, condition, round_number=0, q=None):
        return {'method': method, 'stage': stage, 'condition': condition,
                'round': round_number, 'question_id': q.id if q else None}

    def build(name):
        store = output / name / 'store'
        store.mkdir(parents=True, exist_ok=True)
        method = create(name, store, instance, config.MODEL, Usage(recorder))
        recorder.attach(method.llm, name)
        methods[name] = method
        before = time.monotonic()
        for session in checkpoint.sessions:
            recorder.call(meta(name, 'ingest', 'preparation'), lambda: method.ingest(session))
            print(f'BUILD {name} {session.date}', flush=True)
        recorder.call(meta(name, 'finalize', 'preparation'), method.finalize)
        print(f'READY {name} {time.monotonic()-before:.1f}s', flush=True)
        judge = LLM(config.JUDGE_MODEL, 0.0, output / name / 'judge_log.jsonl')
        judge.chat.callbacks = [Usage(recorder)]
        recorder.attach(judge.chat, name)
        judges[name] = judge

    def answer_phase(names, condition, round_number):
        selector = {'stage': 'answer', 'condition': condition, 'round': round_number}
        before = time.monotonic()

        def one(name, question):
            response = recorder.call(meta(name, 'answer', condition, round_number, question),
                                     lambda: methods[name].answer(question))
            recorder.emit('answer', **meta(name, 'answer', condition, round_number, question),
                          response=response.text, response_meta=response.meta)
            return name, question.id, response.text

        with ThreadPoolExecutor(max_workers=8 * len(names)) as pool:
            futures = [pool.submit(one, name, q) for name in names for q in questions]
            results = [future.result() for future in futures]
        elapsed = time.monotonic() - before
        for name in names:
            rows = summarize(recorder.events, {**selector, 'method': name}, elapsed)
            phases.append(rows)
            print('PHASE', json.dumps(rows), flush=True)
        save(output / 'phases.json', phases)
        return {(name, qid): text for name, qid, text in results}

    async def judge_batch(names, condition, round_number, frozen):
        limits = {name: asyncio.Semaphore(16) for name in names}

        async def one(name, question):
            async with limits[name]:
                verdict, usage = await recorder.acall(meta(name, 'judge', condition, round_number, question),
                    lambda: judges[name].json(JUDGE_SYSTEM, judge_prompt(question, frozen[name, question.id]),
                                             f'{condition}/{round_number}/{question.id}'))
            if verdict.get('verdict') not in ('RIGHT', 'WRONG'):
                raise ValueError(f'Invalid verdict: {verdict}')
            recorder.emit('verdict', **meta(name, 'judge', condition, round_number, question), **verdict)

        await asyncio.gather(*(one(name, q) for name in names for q in questions))

    def judge_phase(names, condition, round_number, frozen):
        before = time.monotonic()
        loop.run_until_complete(judge_batch(names, condition, round_number, frozen))
        elapsed = time.monotonic() - before
        for name in names:
            rows = summarize(recorder.events, {'method': name, 'stage': 'judge',
                'condition': condition, 'round': round_number}, elapsed)
            phases.append(rows)
            print('PHASE', json.dumps(rows), flush=True)
        save(output / 'phases.json', phases)

    failure = None
    try:
        # Verify both paid model endpoints before the larger TiMem preparation.
        build('full_context')
        q = questions[0]
        response = recorder.call(meta('full_context', 'answer', 'warmup', q=q),
                                 lambda: methods['full_context'].answer(q))
        loop.run_until_complete(recorder.acall(meta('full_context', 'judge', 'warmup', q=q),
            lambda: judges['full_context'].json(JUDGE_SYSTEM, judge_prompt(q, response.text), 'warmup')))
        print('AUTH_AND_MODELS_OK', project, config.MODEL, config.JUDGE_MODEL, flush=True)
        for name in ('naive_rag', 'timem'):
            build(name)
            response = recorder.call(meta(name, 'answer', 'warmup', q=q), lambda: methods[name].answer(q))
            loop.run_until_complete(recorder.acall(meta(name, 'judge', 'warmup', q=q),
                lambda: judges[name].json(JUDGE_SYSTEM, judge_prompt(q, response.text), 'warmup')))
        frozen = {}
        for name in METHODS:
            frozen.update(answer_phase([name], 'solitary', 1))
        save(output / 'fixed_judge_inputs.json', [{'method': name, 'question_id': qid, 'response': text}
                                                for (name, qid), text in frozen.items()])
        for name in METHODS:
            judge_phase([name], 'solitary', 1, frozen)
        answer_phase(METHODS, 'concurrent', 1)
        judge_phase(METHODS, 'concurrent', 1, frozen)
        answer_phase(METHODS, 'concurrent', 2)
        judge_phase(METHODS, 'concurrent', 2, frozen)
        for name in reversed(METHODS):
            answer_phase([name], 'solitary', 2)
        for name in reversed(METHODS):
            judge_phase([name], 'solitary', 2, frozen)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)[:500]}
        raise
    finally:
        unchanged = all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                        for path, digest in hashes.items())
        save(output / 'run_summary.json', {'wall_seconds': time.monotonic() - start,
            'complete': failure is None, 'failure': failure, 'source_files_unchanged': unchanged,
            'http_statuses': dict(Counter(str(e['status']) for e in recorder.events if e['kind'] == 'http')),
            'input_tokens': sum(e.get('input_tokens', 0) for e in recorder.events if e['kind'] == 'tokens'),
            'output_tokens': sum(e.get('output_tokens', 0) for e in recorder.events if e['kind'] == 'tokens')})
        for method in methods.values():
            if hasattr(method, 'close'):
                method.close()
        for judge in judges.values():
            loop.run_until_complete(judge.chat.client.aio.aclose())
            judge.chat.client.close()
        loop.close()
        print('PILOT_FINISHED', output, flush=True)


if __name__ == '__main__':
    main()
