"""Deterministic models for protocol tests; vector libraries remain real."""

import importlib
import asyncio
import json
import os
import re
import time
from types import SimpleNamespace
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from experiments import config
from setup_2.methods import METHODS


class Chat(BaseChatModel):
    control: Any = Field(default_factory=dict)

    @property
    def _llm_type(self):
        return 'offline-protocol-model'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = '\n'.join(str(m.content) for m in messages)
        self.control.setdefault('prompts', []).append(prompt)
        content = str(messages[-1].content)
        if content.startswith('MOCK GENERATION\n'):
            response = content.split('\n', 1)[1]
        elif content.startswith('MEMORY CONTEXT:'):
            context, question = content.rsplit('\n\nQUESTION: ', 1)
            city = 'Boston' if '(Asked on 2026-03-' in question else 'Austin'
            response = city if city in context else "I don't know."
        else:
            # Enough actual biography to pass native TiMem's content validators.
            response = ('Alice lives in Boston and works as a designer. She walks by the river '
                        'each morning, enjoys reading history, and discussed her daily routine '
                        'on March 31, 2026. Her favorite local activity is visiting the library.')
        message = AIMessage(content=response, usage_metadata={
            'input_tokens': 7, 'output_tokens': 5, 'total_tokens': 12,
        })
        return ChatResult(generations=[ChatGeneration(message=message)])


class Vectors(Embeddings):
    def __init__(self, control=None):
        self.control = control if control is not None else {}

    def embed_documents(self, texts):
        self.control['embedded_documents'] = self.control.get('embedded_documents', 0) + len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.control['embedded_queries'] = self.control.get('embedded_queries', 0) + 1
        return self._vector(text)

    @staticmethod
    def _vector(text):
        vector = [0.0] * config.EMBED_DIMS
        vector[0] = 1.0
        for offset, month in enumerate(('03', '04', '05', '06'), 1):
            vector[offset] = float(len(re.findall(f'2026-{month}', text)))
        return vector


class AsyncChat(Chat):
    """Simulate a provider transport that belongs to its first running loop."""
    active_loop: Any = None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        current = asyncio.get_running_loop()
        if self.active_loop is None:
            self.active_loop = current
        assert self.active_loop is current
        return self._generate(messages, stop, run_manager, **kwargs)

    async def aclose(self):
        if self.active_loop is not None:
            assert self.active_loop is asyncio.get_running_loop()
        self.control['closed_clients'] = self.control.get('closed_clients', 0) + 1


class Generator:
    def __init__(self, llm, control):
        self.llm, self.control = llm, control
        self.loop = None

    async def _generate(self, layer, contents, period):
        current = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = current
        assert self.loop is current  # Stand-in for an async SDK's loop-bound transport.
        self.control.setdefault('generation', []).append((layer, period))
        return self.llm.invoke([HumanMessage(content='MOCK GENERATION\n' + '\n'.join(contents))]).content

    async def generate_l1_content(self, dialogue):
        return await self._generate('L1', [dialogue], {})

    async def generate_l2_content(self, contents, **period):
        return await self._generate('L2', contents, period)

    async def generate_l3_content(self, contents, **period):
        return await self._generate('L3', contents, period)

    async def generate_l4_content(self, contents, **period):
        return await self._generate('L4', contents, period)

    async def generate_l5_content(self, contents, **period):
        return await self._generate('L5', contents, period)


class Judge:
    def __init__(self, model, path, control=None):
        self.path, self.control = path, control if control is not None else {}

    async def json(self, system, prompt, tag):
        self.control.setdefault('judges', []).append(tag)
        with self.path.open('a') as stream:
            stream.write(json.dumps({'input_tokens': 3, 'output_tokens': 2, 'reasoning_tokens': 0}) + '\n')
        if self.control.get('fail_judge') and tag.endswith(self.control['fail_judge']):
            self.control.pop('fail_judge')
            raise RuntimeError('judge auth failure')
        reference = prompt.split('REFERENCE: ', 1)[1].split('\n', 1)[0]
        answer = prompt.split('SYSTEM ANSWER: ', 1)[1]
        return {'verdict': 'RIGHT' if reference == answer else 'WRONG', 'why': 'offline check'}, \
            SimpleNamespace(input_tokens=3, output_tokens=2, reasoning_tokens=0)


def process_method(name, store, instance, model, usage):
    """A picklable factory tests actual adapters in spawned user processes."""
    module_name, class_name, _ = METHODS[name]
    module = importlib.import_module(module_name)
    control = {}
    module.make_llm = lambda model, usage: Chat(control=control, callbacks=[usage])
    if hasattr(module, 'embeddings'):
        module.embeddings = lambda: Vectors(control)
    method = getattr(module, class_name)(store, instance, model, usage)
    if name == 'timem':
        method._embedder = Vectors(control)
        method._generator = Generator(method.llm, control)
    method.data.set_metadata('worker_pid', os.getpid())
    time.sleep(0.3)
    return method
