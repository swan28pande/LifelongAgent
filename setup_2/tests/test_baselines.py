"""Exercise the actual baseline stores through the common monthly protocol."""

import asyncio
from functools import partial
import importlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from experiments import config
from experiments.core.method import UsageCounter
from experiments.core.types import Instance, Question, Session, Turn
from setup_2.loader import load
from setup_2.methods import METHODS, create
from setup_2.runner import _log, run
from setup_2.tests.fakes import AsyncChat, Chat, Generator, Judge, Vectors, process_method
from setup_2.tests.test_setup import dataset, write


BASELINES = ('timem', 'naive_rag', 'full_context')


@pytest.fixture
def adapters(monkeypatch):
    control = {'constructs': [], 'ingests': [], 'answers': [], 'finalizes': []}
    for name in BASELINES:
        module_name, class_name, _ = METHODS[name]
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, 'make_llm', lambda model, usage: Chat(control=control, callbacks=[usage]))
        if hasattr(module, 'embeddings'):
            monkeypatch.setattr(module, 'embeddings', lambda: Vectors(control))
        cls = getattr(module, class_name)

        def init(self, store, instance, model, usage, original=cls.__init__):
            assert not instance.sessions and not instance.questions
            original(self, store, instance, model, usage)
            control['constructs'].append(self.name)

        def ingest(self, session, original=cls.ingest):
            result = original(self, session)
            control['ingests'].append((self.name, session.date))
            if control.get('fail_ingest') == session.date:
                control.pop('fail_ingest')
                raise RuntimeError('interrupted after baseline writes')
            return result

        def finalize(self, original=cls.finalize):
            original(self)
            control['finalizes'].append(self.name)
            if control.pop('fail_finalize', False):
                raise RuntimeError('interrupted after finalization')

        def answer(self, q, original=cls.answer):
            dates = [s.date for s in self.data.all()]
            control['answers'].append((self.name, q.id, q.asked_on, max(dates, default='')))
            if control.get('fail_answer') == q.id:
                control.pop('fail_answer')
                raise RuntimeError('answer auth failure')
            return original(self, q)

        monkeypatch.setattr(cls, '__init__', init)
        monkeypatch.setattr(cls, 'ingest', ingest)
        monkeypatch.setattr(cls, 'finalize', finalize)
        monkeypatch.setattr(cls, 'answer', answer)
    from setup_2.baselines.TiMem import generator
    monkeypatch.setattr(generator, 'create', lambda model, factory, path: Generator(factory(), control))
    return control


def execute(dataset, output, name, control, **options):
    return run(load(dataset), output, method_name=name,
               judge_factory=partial(Judge, control=control), workers=3, **options)


@pytest.mark.parametrize('name', BASELINES)
def test_baselines_share_cumulative_schedule_and_restore_history(dataset, tmp_path, adapters, name):
    output = tmp_path / name
    result = execute(dataset, output, name, adapters, through='2026-04')
    assert result['method'] == name and result['n'] == 5 and result['accuracy'] == 1
    first = _log(output / 'u1/months/2026-03/answers.jsonl')
    april = _log(output / 'u1/months/2026-04/answers.jsonl')
    assert len(first) == 2 and len(april) == 3
    assert first['u1_q0']['response'] == april['u1_q0']['response'] == 'Boston'
    assert april['u1_q0']['asked_on'] == '2026-03-31'
    assert april['u1_q0']['cutoff'] == '2026-04-30'
    assert april['u1_q1']['response'] == 'Austin'
    assert {item[3] for item in adapters['answers'][:2]} == {'2026-03-31'}
    with sqlite3.connect(output / 'u1/store/baseline.sqlite3') as conn:
        assert conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 4
    old_ingests = list(adapters['ingests'])
    assert execute(dataset, output, name, adapters)['n'] == 13
    assert adapters['ingests'][:len(old_ingests)] == old_ingests
    assert len(adapters['ingests']) == 6
    assert len(adapters['constructs']) == 2
    prompts = len(adapters['prompts'])
    assert execute(dataset, output, name, adapters)['complete']
    assert len(adapters['prompts']) == prompts
    assert len(adapters['constructs']) == 2
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['method_profile'] == METHODS[name][2]


@pytest.mark.parametrize('name', BASELINES)
@pytest.mark.parametrize('stage', ('ingest', 'finalize', 'answer', 'judge'))
def test_baseline_checkpoint_recovery(dataset, tmp_path, adapters, name, stage):
    output = tmp_path / name
    execute(dataset, output, name, adapters, through='2026-03')
    adapters['fail_' + stage] = {'ingest': '2026-04-01', 'finalize': True,
                                'answer': 'u1_qextra', 'judge': 'u1_qextra'}[stage]
    with pytest.raises(RuntimeError):
        execute(dataset, output, name, adapters, through='2026-04')
    count = len(adapters['generation']) if name == 'timem' else 0
    assert execute(dataset, output, name, adapters, through='2026-04')['n'] == 5
    with sqlite3.connect(output / 'u1/store/baseline.sqlite3') as conn:
        assert conn.execute('SELECT count(*),count(DISTINCT id) FROM sessions').fetchone() == (4, 4)
    if name == 'timem':
        if stage in ('answer', 'judge'):
            assert len(adapters['generation']) == count
        with sqlite3.connect(output / 'u1/store/baseline.sqlite3') as conn:
            assert conn.execute('SELECT count(*) FROM nodes WHERE layer="L1"').fetchone()[0] == 4
    if name == 'naive_rag':
        method = create(name, output / 'u1/store', Instance('u1', [], []), 'offline', UsageCounter())
        assert method.index.index.ntotal == 4
    assert not (output / 'u1/checkpoint_backup').exists()
    assert max(s[1] for s in adapters['ingests']) == '2026-04-30'


@pytest.mark.parametrize('name', BASELINES)
def test_empty_months_and_duplicate_ingestion_are_safe(tmp_path, adapters, name):
    store = tmp_path / name
    instance = Instance('same-user', [], [])
    method = create(name, store, instance, 'offline', UsageCounter())
    method.finalize()
    q = Question('q', 'Where do I live?', 'SECRET_REFERENCE', ['SECRET_ACCEPTED'],
                 'value', 'exact', 'fact_current', '2026-03-31')
    assert method.answer(q).text == "I don't know."
    session = Session('s', '2026-03-31', [Turn('Alice', 'I live in Boston.')])
    method.ingest(session)
    with pytest.raises(RuntimeError, match='Finalize'):
        method.answer(q)
    method.finalize()
    calls = len(adapters['prompts'])
    assert method.ingest(session)['added'] is False
    method.finalize()
    assert len(adapters['prompts']) == calls
    method.answer(q)
    assert all('SECRET_REFERENCE' not in p and 'SECRET_ACCEPTED' not in p for p in adapters['prompts'])
    with pytest.raises(ValueError, match='changed'):
        method.ingest(Session('s', '2026-03-31', [Turn('Alice', 'Different observation')]))
    if hasattr(method, 'close'):
        method.close()


def test_timem_handles_cross_month_weeks_and_partial_leap_month(tmp_path, adapters):
    method = create('timem', tmp_path / 'store', Instance('u', [], []), 'offline', UsageCounter())
    for day in ('2026-03-30', '2026-03-31'):
        method.ingest(Session(day, day, [Turn('Alice', 'Boston')]))
    method.finalize()
    march = dict(method.nodes['L5', '2026-03'])
    method.ingest(Session('april', '2026-04-01', [Turn('Alice', 'Austin')]))
    method.finalize()
    assert method.nodes['L5', '2026-03'] == march
    assert method.nodes['L5', '2026-04']['start'] == '2026-04-01'
    assert 'Boston' not in method.nodes['L5', '2026-04']['content']
    week_nodes = [n for n in method.nodes.values() if n['layer'] == 'L4']
    assert len(week_nodes) == 2 and all(n['start'][:7] == n['end'][:7] for n in week_nodes)
    method.ingest(Session('partial', '2028-02-28', [Turn('Alice', 'Austin')]))
    method.finalize()
    assert method.nodes['L5', '2028-02']['end'] == '2028-02-28'
    assert adapters['generation'][-1][1]['month_end'] == '2028-02-28'
    count = len(adapters['generation'])
    method.finalize()
    assert len(adapters['generation']) == count
    method.close()


def test_timem_native_generator_uses_shared_callbacks_and_local_logs(tmp_path, monkeypatch):
    module = importlib.import_module(METHODS['timem'][0])
    control = {}
    monkeypatch.setattr(module, 'make_llm', lambda model, usage: AsyncChat(control=control, callbacks=[usage]))
    monkeypatch.setattr(module, 'embeddings', lambda: Vectors(control))
    usage = UsageCounter()
    method = create('timem', tmp_path / 'store', Instance('u', [], []), 'selected-model', usage)
    method.ingest(Session('s', '2026-03-31', [Turn('Alice', 'I live in Boston and enjoy reading.')]))
    method.finalize()
    assert {n['layer'] for n in method.nodes.values()} == {'L1', 'L2', 'L3', 'L4', 'L5'}
    assert method._generator.llm._llm is None
    assert method._generator.llm.config.model_name == 'selected-model'
    assert usage.snapshot() == {'calls': 5, 'input_tokens': 35, 'output_tokens': 25}
    assert len(list((tmp_path / 'store/logs').glob('prompts_l*.jsonl'))) == 5
    method.close()
    assert control['closed_clients'] == 6  # Five summary clients and the answer client.
    calls = len(control['prompts'])
    documents = control['embedded_documents']
    reloaded = create('timem', tmp_path / 'store', Instance('u', [], []), 'selected-model', usage)
    reloaded.finalize()
    assert len(control['prompts']) == calls and control['embedded_documents'] == documents
    reloaded.close()
    assert control['closed_clients'] == 7


def test_native_generator_failure_is_raised_for_checkpoint_resume():
    from setup_2.baselines.TiMem.generator import CheckpointGenerator
    generator = object.__new__(CheckpointGenerator)
    generator.layer_timeouts = {'l1': 1}
    calls = []

    async def fail():
        calls.append(True)
        raise RuntimeError('expired credentials')

    with pytest.raises(RuntimeError, match='expired credentials'):
        asyncio.run(generator._retry_with_exponential_backoff(fail, 'l1', 'test'))
    assert len(calls) == 1


def test_native_generator_timeout_stops_the_operation(monkeypatch):
    from setup_2.baselines.TiMem import generator as module
    from setup_2.baselines.TiMem.generator import CheckpointGenerator
    generator = object.__new__(CheckpointGenerator)
    generator.layer_timeouts = {'l1': 0.01}
    cancelled = []

    async def stalled_call():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    original_sleep = asyncio.sleep

    async def no_backoff(seconds):
        await original_sleep(0)

    monkeypatch.setattr(module.asyncio, 'sleep', no_backoff)
    with pytest.raises(TimeoutError):
        asyncio.run(generator._retry_with_exponential_backoff(stalled_call, 'l1', 'test'))
    assert cancelled == [True, True, True]


def test_full_context_never_silently_discards_history(tmp_path, adapters, monkeypatch):
    monkeypatch.setattr(config, 'FULL_CONTEXT_MAX_TOKENS', 1)
    method = create('full_context', tmp_path / 'store', Instance('u', [], []), 'offline', UsageCounter())
    method.ingest(Session('s', '2026-03-31', [Turn('Alice', 'Boston')]))
    with pytest.raises(ValueError, match='Full observed history exceeds'):
        method.finalize()
    assert method.dropped == 0


@pytest.mark.parametrize('name', BASELINES)
def test_parallel_user_processes_keep_baseline_stores_independent(dataset, tmp_path, name):
    for filename in ('conversations.json', 'probing_questions.json'):
        data = json.loads((dataset / 'u1' / filename).read_text())
        if filename == 'probing_questions.json':
            data['user_id'] = 'u2'
        else:
            data['name'] = 'Bob'
        write(dataset / 'u2' / filename, data)
    output = tmp_path / name
    result = run(load(dataset), output, method_name=name, through='2026-04', user_workers=2,
                 workers=2, method_factory=partial(process_method, name), judge_factory=Judge)
    assert result['n'] == 10 and result['accuracy'] == 1
    pids = []
    for uid in ('u1', 'u2'):
        with sqlite3.connect(output / uid / 'store/baseline.sqlite3') as conn:
            assert conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 4
            pids.append(json.loads(conn.execute('SELECT value FROM metadata WHERE key="worker_pid"').fetchone()[0]))
        assert json.loads((output / uid / 'state.json').read_text())['completed'] == 2
    assert len(set(pids)) == 2 and os.getpid() not in pids


def test_methods_cannot_resume_each_others_database(dataset, tmp_path, adapters):
    output = tmp_path / 'same-run'
    execute(dataset, output, 'full_context', adapters, through='2026-03')
    with pytest.raises(ValueError, match='method/implementation'):
        execute(dataset, output, 'naive_rag', adapters, through='2026-03')


@pytest.mark.parametrize('name', BASELINES)
def test_baseline_dry_run_loads_only_the_schedule(dataset, name):
    result = subprocess.run([sys.executable, '-m', 'setup_2.run', '--method', name,
                            '--data-dir', str(dataset), '--through', '2026-04', '--dry-run'],
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['method'] == name and plan['answer_instances'] == 5
