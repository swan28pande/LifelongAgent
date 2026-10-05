"""Native TiMem generation with the benchmark model and resumable call failures."""

import asyncio
from contextlib import aclosing
import logging
import random

import httpx
from aiohttp import ClientConnectionError, ClientPayloadError
from google.auth.exceptions import TransportError as AuthTransportError

from llm.base_llm import ModelConfig
from llm.gemini_adapter import GeminiAdapter
from timem.memory.memory_generator import MemoryGenerator
from timem.utils.config_manager import get_config


logger = logging.getLogger(__name__)
SUMMARY_ATTEMPTS = 3
REQUEST_TIMEOUT = 120
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def _retryable(error):
    return (isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError,
                              ClientConnectionError, ClientPayloadError, AuthTransportError))
            or getattr(error, 'code', None) in RETRYABLE_CODES
            or getattr(error, 'status_code', None) in RETRYABLE_CODES)


class SummaryAdapter(GeminiAdapter):
    """Isolate each summary attempt from stale or cancelled model transports."""

    def __init__(self, config, model_factory):
        super().__init__(config)
        self.model_factory = model_factory

    async def chat_stream(self, messages, **kwargs):
        model = self.model_factory()
        # One SDK attempt: retry the whole native operation, including a stream
        # that fails after yielding tokens, without multiplying nested retries.
        try:
            adapter = GeminiAdapter(self.config)
            adapter._llm = model.bind(timeout=REQUEST_TIMEOUT, max_retries=1)
            async with aclosing(adapter.chat_stream(messages, **kwargs)) as stream:
                async for chunk in stream:
                    yield chunk
        finally:
            close = getattr(model, 'aclose', None)
            if close is not None:
                await close()


class CheckpointGenerator(MemoryGenerator):
    def _load_retry_config(self):
        configured = get_config().get('workflows', {}).get('memory_generation', {}).get('layer_timeouts', {})
        defaults = {'l1': 60, 'l2': 120, 'l3': 300, 'l4': 600, 'l5': 900}
        self.layer_timeouts = {layer: configured.get(layer, seconds) for layer, seconds in defaults.items()}

    async def _retry_with_exponential_backoff(self, operation_func, layer, operation_name):
        for attempt in range(1, SUMMARY_ATTEMPTS + 1):
            try:
                return await asyncio.wait_for(operation_func(), timeout=self.layer_timeouts[layer.lower()])
            except Exception as error:
                if not _retryable(error) or attempt == SUMMARY_ATTEMPTS:
                    raise
                delay = 2 ** (attempt - 1) + random.uniform(0, 1)
                logger.warning('%s attempt %s/%s failed (%s); retrying in %.1fs',
                               operation_name, attempt, SUMMARY_ATTEMPTS, type(error).__name__, delay)
                # Native operations allocate fresh content buffers on each call,
                # so an incomplete stream is discarded before the next attempt.
                await asyncio.sleep(delay)

    _retry_with_exponential_backoff_for_streaming = _retry_with_exponential_backoff


def create(model, model_factory, log_directory):
    adapter = SummaryAdapter(ModelConfig(model_name=model, temperature=0.0), model_factory)
    return CheckpointGenerator(llm=adapter, prompt_log_dir=log_directory)
