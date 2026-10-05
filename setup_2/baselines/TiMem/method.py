"""Incremental TiMem hierarchy with the existing dense top-k retrieval profile."""

import asyncio
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session, prompt_for
from .._common import answer_from_context, embeddings, format_session, make_llm
from ..store import Sessions


TIMEM_ROOT = Path(__file__).resolve().parent / 'source'
if str(TIMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEM_ROOT))


class TiMem(MemoryMethod):
    name = 'timem'
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.data = Sessions(store_dir)
        with self.data.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS nodes (
                layer TEXT, key TEXT, content TEXT NOT NULL, signature TEXT NOT NULL,
                start TEXT NOT NULL, end TEXT NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(layer, key))''')
            rows = conn.execute('SELECT * FROM nodes ORDER BY start, layer, key').fetchall()
        self.nodes = {(r['layer'], r['key']): dict(r) for r in rows}
        sessions = self.data.all()
        self.cutoff = max((s.date for s in sessions), default='')
        self.pending = self.data.metadata('finalized_sessions', []) != [s.id for s in sessions]
        self.llm = make_llm(model, usage)
        self._embedder = None
        self._generator = None
        self._loop = None
        # SQLite is authoritative. Rebuilding this isolated search index uses
        # saved vectors, so a restart needs no model calls or remote DB rollback.
        self._qdrant = QdrantClient(':memory:')
        self._collection = 'memories'
        self._qdrant.create_collection(self._collection,
            vectors_config=VectorParams(size=config.EMBED_DIMS, distance=Distance.COSINE))
        for start in range(0, len(rows), 256):
            points = [PointStruct(id=self._id(r['layer'], r['key']),
                                 vector=np.frombuffer(r['vector'], dtype=np.float32).tolist(),
                                 payload=self._payload(r)) for r in rows[start:start + 256]]
            self._qdrant.upsert(self._collection, points=points)

    @staticmethod
    def _id(layer, key):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{layer}/{key}'))

    @staticmethod
    def _payload(node):
        return {key: node[key] for key in ('layer', 'key', 'content', 'start', 'end')}

    def _embeddings(self):
        if self._embedder is None:
            self._embedder = embeddings()
        return self._embedder

    def _gen(self):
        if self._generator is None:
            from .generator import create
            self._generator = create(self.model, lambda: make_llm(self.model, self.usage),
                                     self.store_dir / 'logs')
        return self._generator

    def _generate(self, coroutine):
        # Native async model clients must keep one event loop across months.
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coroutine)

    def _save(self, layer, key, content, signature, start, end):
        from qdrant_client.models import PointStruct
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f'TiMem returned an empty {layer} memory')
        vector = self._embeddings().embed_documents([content])[0]
        node = dict(layer=layer, key=key, content=content, signature=signature,
                    start=start, end=end, vector=np.asarray(vector, dtype=np.float32).tobytes())
        self._qdrant.upsert(self._collection, points=[PointStruct(
            id=self._id(layer, key), vector=vector, payload=self._payload(node))])
        with self.data.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)',
                         tuple(node[k] for k in ('layer', 'key', 'content', 'signature', 'start', 'end', 'vector')))
        self.nodes[layer, key] = node
        return node

    def ingest(self, session: Session) -> dict:
        existing = self.data.get(session.id)
        if existing is not None:
            if existing != session:
                raise ValueError(f'Observed session changed: {session.id}')
            return {'added': False}
        dialogue = format_session(session)
        signature = hashlib.sha256(dialogue.encode()).hexdigest()
        content = self._generate(self._gen().generate_l1_content(dialogue))
        self._save('L1', session.id, content, signature, session.date, session.date)
        self.data.add(session)
        self.cutoff = max(self.cutoff, session.date)
        self.pending = True
        return {'added': True, 'l1_chars': len(content)}

    def _summarize(self, layer, key, children, **period):
        signature = hashlib.sha256(json.dumps(
            {'children': [(n['layer'], n['key'], n['content'], n['start'], n['end']) for n in children],
             'period': period},
            sort_keys=True).encode()).hexdigest()
        existing = self.nodes.get((layer, key))
        if existing and existing['signature'] == signature:
            return existing
        generate = getattr(self._gen(), f'generate_{layer.lower()}_content')
        content = self._generate(generate([n['content'] for n in children], **period))
        return self._save(layer, key, content, signature,
                          min(n['start'] for n in children), max(n['end'] for n in children))

    def finalize(self) -> None:
        sessions = self.data.all()
        daily = defaultdict(list)
        for session in sessions:
            l2 = self._summarize('L2', session.id, [self.nodes['L1', session.id]])
            daily[session.date].append(l2)
        weekly = defaultdict(list)
        for day, children in sorted(daily.items()):
            l3 = self._summarize('L3', day, children, date=day)
            iso_year, week, _ = date.fromisoformat(day).isocalendar()
            # A partial week at a month edge belongs only to that calendar month.
            weekly[(day[:7], iso_year, week)].append(l3)
        monthly = defaultdict(list)
        for (month, iso_year, week), children in sorted(weekly.items()):
            l4 = self._summarize('L4', f'{month}/{iso_year}-W{week:02d}', children,
                                 year=iso_year, week_number=week,
                                 week_start=children[0]['start'], week_end=children[-1]['end'])
            monthly[month].append(l4)
        for month, children in sorted(monthly.items()):
            self._summarize('L5', month, children, year=int(month[:4]), month=int(month[5:]),
                            month_start=children[0]['start'], month_end=children[-1]['end'])
        self.data.set_metadata('finalized_sessions', [s.id for s in sessions])
        self.pending = False

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        points = self._qdrant.query_points(self._collection,
            query=self._embeddings().embed_query(prompt_for(q)), limit=config.TOP_K,
            with_payload=True).points if self.nodes else []
        lines = [f"[{p.payload['layer']}] ({p.payload['start']} to {p.payload['end']}) {p.payload['content']}"
                 for p in points]
        return answer_from_context(self.llm, q, '\n\n'.join(lines) or '(no memories)',
                                   {'hits': len(lines), 'source_cutoff': self.cutoff,
                                    'retrieval_profile': 'hierarchy_dense_top_k',
                                    'layers': [p.payload['layer'] for p in points]})

    def close(self):
        try:
            close_model = getattr(self.llm, 'aclose', None)
            if close_model is not None:
                self._generate(close_model())
            if self._loop is not None:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
        finally:
            if self._loop is not None:
                self._loop.close()
            self._qdrant.close()
