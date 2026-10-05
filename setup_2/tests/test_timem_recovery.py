"""Exercise transient summary failures with real native generation and storage."""

import asyncio
import importlib

import httpx
import pytest
from aiohttp import ClientPayloadError, ServerDisconnectedError
from google.auth.exceptions import RefreshError, TransportError as AuthTransportError
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from experiments.core.method import UsageCounter
from experiments.core.types import Instance, Session, Turn
from setup_2.methods import METHODS, create
from setup_2.tests.fakes import AsyncChat, Vectors


@pytest.fixture
def generator(monkeypatch):
    importlib.import_module(METHODS['timem'][0])  # Locate the copied native packages.
    from setup_2.baselines.TiMem import generator as module

    original_sleep = asyncio.sleep

    async def no_backoff(seconds):
        await original_sleep(0)

    monkeypatch.setattr(module.asyncio, 'sleep', no_backoff)
    result = object.__new__(module.CheckpointGenerator)
    result.layer_timeouts = {'l1': 1}
    return result


@pytest.mark.parametrize('error', (
    TimeoutError('stalled stream'),
    httpx.ReadTimeout('stalled transport'),
    httpx.RemoteProtocolError('disconnected stream'),
    ServerDisconnectedError('disconnected async transport'),
    ClientPayloadError('truncated async stream'),
    AuthTransportError('credential transport unavailable'),
    ClientError(429, {'error': {'message': 'busy'}}),
    ServerError(502, {'error': {'message': 'gateway'}}),
    ServerError(503, {'error': {'message': 'unavailable'}}),
    ServerError(504, {'error': {'message': 'deadline'}}),
))
def test_transient_summary_failure_retries_and_returns_only_completed_result(generator, error):
    calls = []

    async def operation():
        calls.append(True)
        if len(calls) == 1:
            raise error
        return 'complete summary'

    assert asyncio.run(generator._retry_with_exponential_backoff(operation, 'l1', 'summary')) == 'complete summary'
    assert len(calls) == 2


@pytest.mark.parametrize('error', (
    ClientError(400, {'error': {'message': 'invalid request'}}),
    ClientError(401, {'error': {'message': 'expired credentials'}}),
    ClientError(403, {'error': {'message': 'permission denied'}}),
    ValueError('invalid configuration'),
    RefreshError('invalid_grant: expired credentials'),
))
def test_permanent_summary_failure_propagates_without_retry(generator, error):
    calls = []

    async def operation():
        calls.append(True)
        raise error

    with pytest.raises(type(error)) as raised:
        asyncio.run(generator._retry_with_exponential_backoff(operation, 'l1', 'summary'))
    assert raised.value is error and len(calls) == 1


def test_exhausted_transient_failure_is_bounded(generator):
    error = ServerError(504, {'error': {'message': 'deadline'}})
    calls = []

    async def operation():
        calls.append(True)
        raise error

    with pytest.raises(ServerError) as raised:
        asyncio.run(generator._retry_with_exponential_backoff(operation, 'l1', 'summary'))
    assert raised.value is error and len(calls) == 3


class InterruptedStream(AsyncChat):
    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.control.setdefault('request_options', []).append(kwargs)
        attempt = len(self.control['request_options'])
        if attempt == 1:
            yield ChatGenerationChunk(message=AIMessageChunk(content='INCOMPLETE TEXT MUST BE DISCARDED'))
            raise self.control['stream_error']
        yield ChatGenerationChunk(message=AIMessageChunk(
            content='Alice lives in Boston and enjoys reading history by the river each morning.',
            usage_metadata={'input_tokens': 7, 'output_tokens': 5, 'total_tokens': 12}))


@pytest.mark.parametrize('error', (
    httpx.ReadTimeout('connection failed after first token'),
    ServerDisconnectedError('connection failed after first token'),
    ClientPayloadError('stream truncated after first token'),
))
def test_interrupted_native_stream_uses_fresh_clients_and_never_saves_partial_text(tmp_path, monkeypatch, generator, error):
    module = importlib.import_module(METHODS['timem'][0])
    control, models = {'stream_error': error}, []

    def model_factory(model, usage):
        result = InterruptedStream(control=control, callbacks=[usage])
        models.append(result)
        return result

    monkeypatch.setattr(module, 'make_llm', model_factory)
    monkeypatch.setattr(module, 'embeddings', lambda: Vectors(control))
    usage = UsageCounter()
    method = create('timem', tmp_path / 'store', Instance('u', [], []), 'selected-model', usage)
    try:
        method.ingest(Session('s', '2026-03-31', [Turn('Alice', 'I live in Boston.')]))
        node = method.nodes['L1', 's#0000']
        assert 'Boston' in node['content'] and 'INCOMPLETE' not in node['content']
        assert len(models) == 3 and len({id(m) for m in models}) == 3
        assert control['closed_clients'] == 2
        assert all(o['timeout'] == 120 and o['max_retries'] == 1 for o in control['request_options'])
        assert usage.snapshot() == {'calls': 1, 'input_tokens': 7, 'output_tokens': 5}
        with method.data.connect() as conn:
            assert conn.execute('SELECT content FROM nodes').fetchone()[0] == node['content']
    finally:
        method.close()
    assert control['closed_clients'] == 3
