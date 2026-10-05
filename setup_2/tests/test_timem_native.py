"""Native TiMem fragments, summary history, complexity-aware recall and Gemini output checks."""

import asyncio
import importlib

import pytest
from langchain_core.messages import AIMessageChunk

from experiments.core.method import UsageCounter
from experiments.core.types import Instance, Question, Session, Turn
from setup_2.methods import METHODS, create
from setup_2.tests.fakes import Chat, Generator, Vectors


@pytest.fixture
def timem(tmp_path, monkeypatch):
    control = {}
    module = importlib.import_module(METHODS['timem'][0])
    monkeypatch.setattr(module, 'make_llm', lambda model, usage: Chat(control=control, callbacks=[usage]))
    monkeypatch.setattr(module, 'embeddings', lambda: Vectors(control))
    from setup_2.baselines.TiMem import generator
    monkeypatch.setattr(generator, 'create', lambda model, factory, path: Generator(factory(), control))
    method = create('timem', tmp_path / 'store', Instance('u', [], []), 'offline', UsageCounter())
    yield method, control
    method.close()


def question(text='Where do I live?'):
    return Question('q', text, 'Boston', ['Boston'], 'value', 'exact', 'fact_current', '2026-03-31')


def test_turn_pairs_become_fragments_with_session_history(timem):
    method, control = timem
    turns = [Turn('Alice', f'turn {i}') for i in range(5)]
    assert method.ingest(Session('s', '2026-03-02', turns))['l1_fragments'] == 3
    keys = sorted(k for layer, k in method.nodes if layer == 'L1')
    assert keys == ['s#0000', 's#0001', 's#0002']
    assert method.nodes['L1', 's#0002']['content'] == 'Alice: turn 4'  # odd final turn is kept
    previous = [period['previous_content'] for layer, period in control['generation'] if layer == 'L1']
    assert previous[0] is None
    assert previous[1] == '[2026-03-02] Alice: turn 0\nAlice: turn 1'
    assert previous[2].count('[2026-03-02]') == 2


def test_summaries_receive_three_most_recent_earlier_summaries(timem):
    method, control = timem
    for day in ('2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06'):
        method.ingest(Session(day, day, [Turn('Alice', f'note {day}')]))
    method.finalize()
    l2 = [period['previous_content'] for layer, period in control['generation'] if layer == 'L2']
    assert l2[0] is None
    last = l2[-1]
    assert last.count('Session:') == 3
    assert last.index('2026-03-05') < last.index('2026-03-03')  # newest first
    assert '2026-03-02' not in last
    generated = len(control['generation'])
    method.finalize()
    assert len(control['generation']) == generated  # unchanged history does not regenerate


@pytest.mark.parametrize('complexity,levels', [
    (0, {'L1', 'L2', 'L5'}),
    (1, {'L1', 'L2', 'L3', 'L5'}),
    (2, {'L1', 'L2', 'L3', 'L4', 'L5'}),
])
def test_recall_collects_the_strategy_layers_and_applies_the_refiner(timem, complexity, levels):
    method, control = timem
    method.ingest(Session('a', '2026-03-02', [Turn('Alice', 'I live in Boston.'), Turn('Bob', 'Nice.')]))
    method.ingest(Session('b', '2026-03-10', [Turn('Alice', 'Still in Boston.')]))
    method.finalize()
    control['complexity'] = complexity
    answer = method.answer(question())
    assert answer.meta['strategy'] == ('simple', 'hybrid', 'complex')[complexity]
    assert set(answer.meta['layers']) == levels
    assert answer.meta['candidates'] == control['refined'][-1] and not answer.meta['refiner_failed']
    assert answer.text == 'Boston'
    control['keep_ids'] = [1]  # refiner keeps one candidate; the rest never reach the answer prompt
    gated = method.answer(question())
    assert gated.meta['hits'] == 1 and len(gated.meta['layers']) == 1


def test_bm25_keywords_lift_matching_fragments(timem):
    method, control = timem
    method.ingest(Session('a', '2026-03-02', [Turn('Alice', 'My cat is called Luna.')]))
    method.ingest(Session('b', '2026-03-03', [Turn('Alice', 'I went running today.')]))
    method.finalize()
    from setup_2.baselines.TiMem.method import STRATEGIES
    cfg = method._recall_config()
    ranked = method._rank_l1('pet', ['luna'], cfg['retrieval_strategies'][STRATEGIES[0]], cfg)
    scores = {m['key']: m['bm25_score'] for m in ranked}
    assert scores['a#0000'] > 0 and scores['b#0000'] == 0
    assert [m['created_at'] for m in ranked] == sorted(m['created_at'] for m in ranked)


class _Stream(Chat):
    chunks: list = []
    finish: str = 'STOP'

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGenerationChunk
        for i, text in enumerate(self.chunks):
            meta = {'finish_reason': self.finish} if i == len(self.chunks) - 1 else {}
            yield ChatGenerationChunk(message=AIMessageChunk(content=[{'type': 'text', 'text': text}],
                                                             response_metadata=meta))


def _stream(chunks, finish='STOP'):
    from llm.base_llm import Message, MessageRole, ModelConfig
    from llm.gemini_adapter import GeminiAdapter
    adapter = GeminiAdapter(ModelConfig(model_name='offline', temperature=0.0))
    adapter._llm = _Stream(chunks=chunks, finish=finish)

    async def collect():
        return ''.join([c async for c in adapter.chat_stream([Message(role=MessageRole.USER, content='x')])])
    return asyncio.run(collect())


def test_stream_keeps_spaces_between_chunks(timem):
    assert _stream(['LGBTQ ', 'rights and ', 'poetry']) == 'LGBTQ rights and poetry'


@pytest.mark.parametrize('chunks,finish', [(['partial summ'], 'MAX_TOKENS'), (['   '], 'STOP')])
def test_stream_rejects_truncated_or_empty_output(timem, chunks, finish):
    with pytest.raises(RuntimeError, match='unusable output'):
        _stream(chunks, finish)
