"""Mem0 open source (mem0ai 2.2.0): additive LLM fact extraction over a vector store.

Mem0 2.x appends newly extracted facts (exact duplicates skipped) and does not UPDATE or DELETE
existing memories, unlike the ECAI 2025 paper's algorithm. Each session's date is passed to the
extraction prompt as its Observation Date, which Mem0 OSS otherwise sets to today's system date.

Configured as in experiments/methods/mem0_oss.py: Gemini on Vertex for extraction and answering,
the shared Nomic embedder, FAISS, top-k search and the shared answer prompt. Each session is added
once, in date order; Mem0 updates its memories incrementally, so finalization has nothing to build.
All Mem0 state (FAISS index and history database) lives in the user's store, which the monthly
runner backs up and restores as a unit.
"""

import os
import threading
import time

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session, prompt_for
from experiments.methods._common import mem0_observation_date
from .._common import answer_from_context, make_llm
from ..store import Sessions

os.environ.setdefault('MEM0_TELEMETRY', 'False')

# Answer threads share one Nomic model, whose remote code corrupts tensors under concurrent calls.
_EMBED_LOCK = threading.Lock()


class Mem0(MemoryMethod):
    name = 'mem0'
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from mem0 import Memory
        from memory_v3.agent import _vertex_project

        self.data = Sessions(store_dir)
        self.user_id = instance.id
        mem0_dir = store_dir / 'mem0'
        mem0_dir.mkdir(parents=True, exist_ok=True)
        self.memory = Memory.from_config({
            'llm': {'provider': 'gemini', 'config': {
                # mem0's default max_tokens counts Gemini's thinking tokens and truncates the JSON.
                'model': model, 'temperature': 0.0, 'max_tokens': 32000, 'vertexai': True,
                'project': _vertex_project(), 'location': os.getenv('GOOGLE_CLOUD_LOCATION', 'global'),
            }},
            'embedder': {'provider': 'huggingface', 'config': {
                'model': config.EMBED_MODEL, 'embedding_dims': config.EMBED_DIMS,
                'model_kwargs': {'trust_remote_code': True},
            }},
            'vector_store': {'provider': 'faiss', 'config': {
                'path': str(mem0_dir / 'faiss'), 'collection_name': 'memories',
                'embedding_model_dims': config.EMBED_DIMS,
            }},
            'history_db_path': str(mem0_dir / 'history.db'),
        })
        embed = self.memory.embedding_model.embed

        def locked_embed(*args, **kwargs):
            with _EMBED_LOCK:
                return embed(*args, **kwargs)

        self.memory.embedding_model.embed = locked_embed
        self.pending = False
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        existing = self.data.get(session.id)
        if existing is not None:
            if existing != session:
                raise ValueError(f'Observed session changed: {session.id}')
            return {'added': False}
        messages = []
        for t in session.turns:
            role = t.speaker if t.speaker in ('user', 'assistant') else 'user'
            name = '' if t.speaker in ('user', 'assistant') else f'{t.speaker}: '
            messages.append({'role': role, 'content': f'[{session.date}] {name}{t.text}'})
        for attempt in range(3):
            try:
                with mem0_observation_date(session.date):
                    result = self.memory.add(messages, user_id=self.user_id, metadata={'date': session.date})
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        # Recorded only after Mem0 accepted the session; a failed month is rolled back as a whole.
        self.data.add(session)
        self.pending = True
        return {'added': True, 'memories_changed': len(result.get('results', []))}

    def finalize(self) -> None:
        self.pending = False

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        hits = self.memory.search(prompt_for(q), filters={'user_id': self.user_id}, top_k=config.TOP_K)
        hits = hits.get('results', []) if isinstance(hits, dict) else hits
        lines = [f"- [{(h.get('metadata') or {}).get('date', '?')}] {h['memory']}" for h in hits]
        return answer_from_context(self.llm, q, '\n'.join(lines) or '(no memories)', {'hits': len(hits)})
