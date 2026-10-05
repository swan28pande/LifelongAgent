"""Exercise the monthly protocol through persistent memory and judge interfaces."""

import asyncio
import copy
import datetime as dt
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.core.types import Answer, prompt_for
from setup_2.loader import describe, load
from setup_2.runner import _log, run


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def dataset(tmp_path):
    base = tmp_path / 'data'
    dates = ['2026-03-03', '2026-03-31', '2026-04-01', '2026-04-30', '2026-05-31', '2026-06-12']
    sessions = {date: {'turns': [{'speaker': 'user', 'text': 'Boston' if date[:7] == '2026-03' else 'Austin'}]}
                for date in reversed(dates)}
    probes = []
    for index, date in enumerate(('2026-03-31', '2026-04-30', '2026-05-31')):
        viewpoint = (dt.date.fromisoformat(date) - dt.date(2026, 3, 1)).days + 1
        rows = [{
            'id': f'u1_q{index}', 'original_id': f'source_{index}', 'type': 'fact_current',
            'question': 'Where do I currently live?', 'answer': 'Boston' if index == 0 else 'Austin',
            'accept': ['Boston' if index == 0 else 'Austin'], 'answer_type': 'value', 'grading': 'exact',
            'viewpoint_day': viewpoint, 'probe_date': date, 'probe_month': date[:7],
            'capability': 'knowledge_update', 'evidence_days': [1], 'is_retrospective': False, 'tags': [],
        }]
        if index == 0:
            second = copy.deepcopy(rows[0]); second['id'] = 'u1_qextra'; rows.append(second)
        probes.append({'probe_date': date, 'probe_month': date[:7], 'viewpoint_day': viewpoint,
                       'num_questions': len(rows), 'questions': rows})
    write(base / 'u1/conversations.json', {'name': 'Alice', 'sessions': sessions})
    write(base / 'u1/probing_questions.json', {'user_id': 'u1', 'start_date': '2026-03-01',
          'num_days': 104, 'num_probes': 3, 'total_questions': 4, 'probes': probes})
    return base


class Usage:
    def __init__(self):
        self.calls = 0

    def tick(self):
        self.calls += 1

    def snapshot(self):
        return {'calls': self.calls, 'input_tokens': self.calls * 3, 'output_tokens': self.calls * 2}


@pytest.fixture
def harness(tmp_path):
    control = {'ingests': [], 'answers': [], 'finalizes': [], 'judges': [], 'constructs': 0}

    class Memory:
        def __init__(self, store, instance, model, usage):
            control['constructs'] += 1
            assert not instance.sessions and not instance.questions  # No future content at construction.
            self.store, self.usage = store, usage
            store.mkdir(exist_ok=True)
            self.db = store / 'memories.db'
            with sqlite3.connect(self.db) as conn:
                conn.execute('CREATE TABLE IF NOT EXISTS conversations(date TEXT, value TEXT)')

        def rows(self):
            with sqlite3.connect(self.db) as conn:
                return conn.execute('SELECT date,value FROM conversations ORDER BY date').fetchall()

        def ingest(self, session):
            self.usage.tick()
            control['ingests'].append(session.date)
            with sqlite3.connect(self.db) as conn:
                conn.execute('INSERT INTO conversations VALUES (?,?)', (session.date, session.turns[0].text))
            vectors = self.store / 'faiss_conversations' / 'saved_vectors.txt'
            vectors.parent.mkdir(exist_ok=True)
            with vectors.open('a') as f:
                f.write(session.date + '\n')
            if control.get('fail_ingest') == session.date:
                control.pop('fail_ingest')
                raise RuntimeError('interrupted after a partial store write')
            return {'added': 1}

        def finalize(self):
            self.usage.tick()
            control['finalizes'].append(self.rows()[-1][0])
            (self.store / 'summaries.json').write_text(json.dumps(self.rows()))
            if control.pop('fail_finalize', False):
                (self.store / 'partial_distillation').write_text('unfinished')
                raise RuntimeError('interrupted during distillation')

        def answer(self, q):
            self.usage.tick()
            rows = self.rows()
            control['answers'].append((rows[-1][0], q.id, prompt_for(q)))
            if control.get('fail_answer') == q.id:
                control.pop('fail_answer')
                raise RuntimeError('answer auth failure')
            historical = [value for date, value in rows if date <= q.asked_on]
            return Answer(historical[-1], {'visible_dates': [date for date, _ in rows],
                          'tool_calls': [{'tool': 'search_memories', 'args': {'end_date': q.asked_on}}]})

    class Judge:
        def __init__(self, model, path):
            self.path = path

        async def json(self, system, prompt, tag):
            control['judges'].append(tag)
            await asyncio.sleep(0)
            with self.path.open('a') as f:
                f.write(json.dumps({'input_tokens': 3, 'output_tokens': 2, 'reasoning_tokens': 0}) + '\n')
            if control.get('fail_judge') and tag.endswith(control['fail_judge']):
                control.pop('fail_judge')
                raise RuntimeError('judge auth failure')
            if control.pop('malformed_judge', False):
                return {'verdict': 'MAYBE'}, SimpleNamespace(input_tokens=3, output_tokens=2)
            reference = prompt.split('REFERENCE: ', 1)[1].split('\n', 1)[0]
            response = prompt.split('SYSTEM ANSWER: ', 1)[1]
            return {'verdict': 'RIGHT' if reference == response else 'WRONG', 'why': 'offline comparison'}, \
                SimpleNamespace(input_tokens=3, output_tokens=2, reasoning_tokens=0)

    def execute(timelines, *, workers=1, **kwargs):
        return run(timelines, tmp_path / 'run', workers=workers, method_factory=Memory,
                   judge_factory=Judge, usage_factory=Usage, **kwargs)

    return SimpleNamespace(control=control, execute=execute, output=tmp_path / 'run')


def test_loader_cumulative_dates_and_partial_month(dataset):
    timeline = load(dataset)[0]
    assert [c.cutoff for c in timeline.checkpoints] == ['2026-03-31', '2026-04-30', '2026-05-31', '2026-06-12']
    assert [len(c.questions) for c in timeline.checkpoints] == [2, 3, 4, 4]
    assert [s.date for c in timeline.checkpoints for s in c.sessions] == sorted(
        json.loads((dataset / 'u1/conversations.json').read_text())['sessions'])
    original = timeline.checkpoints[0].questions[0]
    assert timeline.checkpoints[-1].questions[0] is original
    assert original.asked_on == '2026-03-31' and original.answer == 'Boston'
    assert describe([timeline])['answer_instances'] == 13
    assert timeline.checkpoints[-1].new_question_ids == ()


def test_monthly_memory_grows_without_future_information(dataset, harness):
    result = harness.execute(load(dataset))
    assert result['complete'] and result['n'] == 13 and result['accuracy'] == 1
    assert len(harness.control['ingests']) == 6 and len(harness.control['finalizes']) == 4
    assert result['judge_usage']['calls'] == 13
    assert harness.control['constructs'] == 1
    first = _log(harness.output / 'u1/months/2026-03/answers.jsonl')
    april = _log(harness.output / 'u1/months/2026-04/answers.jsonl')
    assert set(first) == {'u1_q0', 'u1_qextra'}
    assert max(first['u1_q0']['meta']['visible_dates']) == '2026-03-31'
    assert first['u1_q0']['response'] == april['u1_q0']['response'] == 'Boston'
    assert april['u1_q0']['asked_on'] == '2026-03-31' and april['u1_q0']['cutoff'] == '2026-04-30'
    assert april['u1_q1']['response'] == 'Austin'
    assert max(april['u1_q0']['meta']['visible_dates']) == '2026-04-30'
    with sqlite3.connect(harness.output / 'u1/store/memories.db') as conn:
        assert conn.execute('SELECT count(*) FROM conversations').fetchone()[0] == 6
    assert not (harness.output / 'u1/checkpoint_backup').exists()
    calls = copy.deepcopy(harness.control)
    assert harness.execute(load(dataset))['n'] == 13
    assert harness.control == calls  # A completed resume makes no new model calls.


def test_extend_a_short_run_without_reingesting(dataset, harness):
    assert harness.execute(load(dataset), through='2026-03')['n'] == 2
    assert len(harness.control['ingests']) == 2
    assert harness.execute(load(dataset), through='2026-04')['n'] == 5
    assert len(harness.control['ingests']) == 4
    assert harness.execute(load(dataset))['n'] == 13
    assert len(harness.control['ingests']) == 6


def test_partial_ingestion_rolls_back_both_store_and_receipts(dataset, harness):
    harness.control['fail_ingest'] = '2026-04-01'
    with pytest.raises(RuntimeError, match='partial store write'):
        harness.execute(load(dataset))
    state = json.loads((harness.output / 'u1/state.json').read_text())
    assert state == {'completed': 1, 'phase': 'building', 'month': '2026-04'}
    assert not (harness.output / 'u1/months/2026-04/answers.jsonl').exists()
    harness.execute(load(dataset))
    with sqlite3.connect(harness.output / 'u1/store/memories.db') as conn:
        assert conn.execute('SELECT count(*),count(DISTINCT date) FROM conversations').fetchone() == (6, 6)
    vectors = (harness.output / 'u1/store/faiss_conversations/saved_vectors.txt').read_text().splitlines()
    assert len(vectors) == len(set(vectors)) == 6  # Separate index writes were rolled back too.
    assert set(_log(harness.output / 'u1/months/2026-04/ingest.jsonl')) == {'2026-04-01', '2026-04-30'}
    assert harness.control['ingests'].count('2026-03-03') == 1
    assert harness.control['ingests'].count('2026-04-01') == 2


def test_interrupted_distillation_is_rebuilt_before_questions(dataset, harness):
    harness.control['fail_finalize'] = True
    with pytest.raises(RuntimeError, match='distillation'):
        harness.execute(load(dataset))
    assert harness.control['answers'] == []
    harness.execute(load(dataset))
    assert not (harness.output / 'u1/store/partial_distillation').exists()
    with sqlite3.connect(harness.output / 'u1/store/memories.db') as conn:
        assert conn.execute('SELECT count(*) FROM conversations').fetchone()[0] == 6


@pytest.mark.parametrize('stage', ['answer', 'judge'])
def test_resume_missing_evaluations_keeps_checkpoint_memory_fixed(dataset, harness, stage):
    harness.control['fail_' + stage] = 'u1_qextra'
    with pytest.raises(RuntimeError, match='auth failure'):
        harness.execute(load(dataset))
    assert harness.control['ingests'] == ['2026-03-03', '2026-03-31']
    assert json.loads((harness.output / 'u1/state.json').read_text())['phase'] == 'evaluating'
    harness.execute(load(dataset))
    assert len(harness.control['ingests']) == 6 and len(harness.control['finalizes']) == 4
    march_success = [r for r in harness.control['answers'] if r[:2] == ('2026-03-31', 'u1_q0')]
    assert len(march_success) == 1


def test_invalid_judge_result_stays_pending(dataset, harness):
    harness.control['malformed_judge'] = True
    with pytest.raises(RuntimeError, match='Invalid judge verdict'):
        harness.execute(load(dataset))
    assert harness.execute(load(dataset))['accuracy'] == 1


def test_changed_inputs_or_models_refuse_existing_store(dataset, harness):
    harness.execute(load(dataset), through='2026-03')
    with pytest.raises(ValueError, match='Inputs/models/protocol changed'):
        harness.execute(load(dataset), model='different-model')
    p = dataset / 'u1/probing_questions.json'
    data = json.loads(p.read_text()); data['probes'][0]['questions'][0]['answer'] = 'corrected'
    write(p, data)
    with pytest.raises(ValueError, match='Inputs/models/protocol changed'):
        harness.execute(load(dataset))
    assert len(harness.control['ingests']) == 2


def test_missing_completed_answers_never_replay_against_future_store(dataset, harness):
    harness.execute(load(dataset))
    (harness.output / 'u1/months/2026-03/answers.jsonl').unlink()
    count = len(harness.control['answers'])
    with pytest.raises(RuntimeError, match='Unexpected answer/grade IDs|missing results'):
        harness.execute(load(dataset))
    assert len(harness.control['answers']) == count
    assert not json.loads((harness.output / 'summary.json').read_text())['complete']


def test_missing_state_refuses_store_reuse(dataset, harness):
    harness.execute(load(dataset))
    (harness.output / 'u1/state.json').unlink()
    with pytest.raises(RuntimeError, match='Missing checkpoint state'):
        harness.execute(load(dataset))


def test_separate_users_get_independent_stores(dataset, harness):
    for name in ('conversations.json', 'probing_questions.json'):
        data = json.loads((dataset / 'u1' / name).read_text())
        if name == 'probing_questions.json': data['user_id'] = 'u2'
        else: data['name'] = 'Bob'
        write(dataset / 'u2' / name, data)
    result = harness.execute(load(dataset))
    assert result['n'] == 26
    for uid in ('u1', 'u2'):
        with sqlite3.connect(harness.output / uid / 'store/memories.db') as conn:
            assert conn.execute('SELECT count(*) FROM conversations').fetchone()[0] == 6


def test_empty_months_do_not_reset_the_cumulative_set(dataset):
    p = dataset / 'u1/probing_questions.json'
    data = json.loads(p.read_text()); data['probes'].pop(1)
    data['num_probes'] = 2; data['total_questions'] = 3
    write(p, data)
    timeline = load(dataset)[0]
    assert [len(c.questions) for c in timeline.checkpoints] == [2, 2, 3, 3]
    assert timeline.checkpoints[1].new_question_ids == ()


@pytest.mark.parametrize('change', ['duplicate_id', 'wrong_date'])
def test_invalid_probe_metadata_is_rejected(dataset, change):
    p = dataset / 'u1/probing_questions.json'
    data = json.loads(p.read_text())
    q = data['probes'][1]['questions'][0]
    q['id' if change == 'duplicate_id' else 'probe_date'] = 'u1_q0' if change == 'duplicate_id' else '2026-03-31'
    write(p, data)
    with pytest.raises(ValueError): load(dataset)


def test_torn_last_record_recovers_but_committed_corruption_fails(tmp_path):
    p = tmp_path / 'log.jsonl'; p.write_bytes(b'{"id":"saved"}\n{"id":')
    assert _log(p) == {'saved': {'id': 'saved'}}
    assert p.read_bytes().endswith(b'\n')
    p.write_bytes(b'not json\n')
    with pytest.raises(ValueError): _log(p)


def test_a_second_process_cannot_use_the_same_run(dataset, harness):
    from setup_2.runner import _lock
    with _lock(harness.output):
        with pytest.raises(RuntimeError, match='Another process'):
            harness.execute(load(dataset))


def test_rolling_back_an_interrupted_copy_can_be_retried(dataset, harness):
    harness.control['fail_ingest'] = '2026-04-01'
    with pytest.raises(RuntimeError): harness.execute(load(dataset))
    # Simulate a kill after deleting the damaged store, part-way through recovery.
    import shutil
    shutil.rmtree(harness.output / 'u1/store')
    assert harness.execute(load(dataset))['n'] == 13


def test_answer_parallelism_preserves_the_checkpoint_barrier(dataset, harness):
    result = harness.execute(load(dataset), workers=3)
    assert result['n'] == 13
    for month, cutoff in [('2026-03', '2026-03-31'), ('2026-04', '2026-04-30')]:
        answers = _log(harness.output / f'u1/months/{month}/answers.jsonl')
        assert all(max(a['meta']['visible_dates']) == cutoff for a in answers.values())


def test_dry_run_never_constructs_agent_or_creates_results(dataset):
    result = subprocess.run([sys.executable, '-m', 'setup_2.run', '--data-dir', str(dataset),
                            '--users', 'u1', '--through', '2026-04', '--dry-run'],
                            capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['answer_instances'] == 5 and data['sessions'] == 4


def test_existing_adapter_enables_both_summaries_and_distillation(tmp_path, monkeypatch):
    pytest.importorskip('langchain_core')
    pytest.importorskip('langgraph')
    from memory_v4 import agent
    from setup_2.runner import _method
    from experiments.core.types import Instance
    calls = []

    class Agent:
        def __init__(self, **kwargs): pass
        def build_summaries(self, **kwargs): calls.append(kwargs)
        def chat_with_trace(self, prompt):
            calls.append(prompt)
            return {'answer': 'Boston', 'tool_calls': []}

    monkeypatch.setattr(agent, 'AgenticMemoryAgent', Agent)
    method = _method(tmp_path, Instance('u1', [], [], 'Alice'), 'existing-model', Usage())
    method.finalize()
    from experiments.core.types import Question
    q = Question('q', 'Where do I currently live?', 'Boston', ['Boston'], 'value', 'exact',
                 'fact_current', '2026-03-31')
    assert method.answer(q).text == 'Boston'
    assert calls == [{'force': True, 'distill': True}, '(Asked on 2026-03-31) Where do I currently live?']
