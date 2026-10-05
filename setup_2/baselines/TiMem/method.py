"""Incremental TiMem: native 2-turn L1 fragments, five-level hierarchy and the paper's recall.

Memory follows the upstream generation workflow: every two turns form an L1 fragment written with
the three previous fragments of its session as context, and each L2-L5 summary is written from its
children plus the three most recent earlier summaries of the same level.

Recall follows the upstream complexity-aware workflow (timem/workflows/memory_retrieval.py): the
native planner labels the question simple/hybrid/complex and extracts keywords; L1 is ranked by
0.9 x normalized dense similarity + 0.1 x normalized BM25 (top 40 dense, BM25 over all L1, keep 20,
time-ordered); parents are collected bottom-up per the strategy's layers and limits; the native
memory refiner gates the candidates with the strategy prompt. Storage stays local to this adapter.
"""

import asyncio
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import uuid

import numpy as np

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session, prompt_for
from .._common import answer_from_context, embeddings, make_llm
from ..store import Sessions


TIMEM_ROOT = Path(__file__).resolve().parent / 'source'
if str(TIMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEM_ROOT))
os.environ.setdefault('TIMEM_DATASET_PROFILE', 'default')
# The native config manager resolves paths from the working directory and otherwise silently falls back
# to built-in strategies with different layers and limits; pin the source's own recall configuration.
os.environ.setdefault('TIMEM_RETRIEVAL_CONFIG_PATH',
                      str(TIMEM_ROOT / 'config' / 'datasets' / 'default' / 'retrieval_config.yaml'))

LAYERS = ('L1', 'L2', 'L3', 'L4', 'L5')
STRATEGIES = {0: 'simple', 1: 'hybrid', 2: 'complex'}
HISTORY = 3
# Complex recall ranks by fused score times a level weight (upstream complex_retriever).
IMPORTANCE = {'L1': 1.0, 'L5': 0.8, 'L2': 0.7, 'L3': 0.6, 'L4': 0.5}


def _fragments(session: Session) -> list[str]:
    """Upstream pairs consecutive turns; an odd final turn becomes its own fragment."""
    turns = session.turns
    return ['\n'.join(f'{t.speaker}: {t.text}' for t in turns[i:i + 2]) for i in range(0, len(turns), 2)]


def _week_key(day: str) -> str:
    iso_year, week, _ = date.fromisoformat(day).isocalendar()
    # A partial week at a month edge belongs only to that calendar month.
    return f'{day[:7]}/{iso_year}-W{week:02d}'


def _bm25(documents: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Upstream BM25 (bottom_up_retriever_base._calculate_bm25_scores), scored over every document."""
    tokens = [re.findall(r'\b\w+\b', d.lower()) for d in documents]
    freqs = [Counter(t) for t in tokens]
    lengths = [len(t) for t in tokens]
    avgdl = sum(lengths) / len(lengths) if lengths else 0
    df = Counter(word for f in freqs for word in f)
    idf = {word: math.log(len(documents) / count) for word, count in df.items()}
    words = re.findall(r'\b\w+\b', query.lower())
    scores = []
    for f, length in zip(freqs, lengths):
        score = 0.0
        for word in words:
            if word in f:
                tf = f[word]
                score += idf.get(word, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avgdl))
        scores.append(score)
    return scores


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
        self._planner = None
        self._refiner = None
        self._recall_llm = None
        self._recall_lock = threading.Lock()
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

    # ---------------------------------------------------------------- memory generation

    def ingest(self, session: Session) -> dict:
        existing = self.data.get(session.id)
        if existing is not None:
            if existing != session:
                raise ValueError(f'Observed session changed: {session.id}')
            return {'added': False}
        written = []
        for index, fragment in enumerate(_fragments(session)):
            # Upstream L1 context: the session's three most recent fragments, oldest first.
            previous = '\n'.join(f'[{session.date}] {c}' for c in written[-HISTORY:]) or None
            content = self._generate(self._gen().generate_l1_content(fragment, previous))
            signature = hashlib.sha256(json.dumps([fragment, previous]).encode()).hexdigest()
            self._save('L1', f'{session.id}#{index:04d}', content, signature, session.date, session.date)
            written.append(content)
        self.data.add(session)
        self.cutoff = max(self.cutoff, session.date)
        self.pending = True
        return {'added': True, 'l1_fragments': len(written), 'l1_chars': sum(map(len, written))}

    def _history(self, earlier):
        """Upstream L2-L5 context: three most recent earlier summaries of the level, newest first."""
        recent = sorted(earlier, key=lambda n: (n['start'], n['key']), reverse=True)[:HISTORY]
        text = '\n\n'.join(f"[{n['start']}] Session: {n['content']}" for n in recent)
        return recent, text or None

    def _summarize(self, layer, key, children, earlier, **period):
        history, previous = self._history(earlier)
        signature = hashlib.sha256(json.dumps(
            {'children': [(n['layer'], n['key'], n['content'], n['start'], n['end']) for n in children],
             'history': [(n['key'], n['signature']) for n in history],
             'period': period},
            sort_keys=True).encode()).hexdigest()
        existing = self.nodes.get((layer, key))
        if existing and existing['signature'] == signature:
            return existing
        generate = getattr(self._gen(), f'generate_{layer.lower()}_content')
        content = self._generate(generate([n['content'] for n in children], previous_content=previous, **period))
        return self._save(layer, key, content, signature,
                          min(n['start'] for n in children), max(n['end'] for n in children))

    def _fragments_of(self, session_id):
        prefix = f'{session_id}#'
        return [n for (layer, key), n in sorted(self.nodes.items())
                if layer == 'L1' and key.startswith(prefix)]

    def finalize(self) -> None:
        sessions = self.data.all()
        done = defaultdict(list)
        daily = defaultdict(list)
        for session in sessions:
            children = self._fragments_of(session.id)
            if not children:
                continue
            l2 = self._summarize('L2', session.id, children, list(done['L2']))
            done['L2'].append(l2)
            daily[session.date].append(l2)
        weekly = defaultdict(list)
        for day, children in sorted(daily.items()):
            l3 = self._summarize('L3', day, children, list(done['L3']), date=day)
            done['L3'].append(l3)
            weekly[_week_key(day)].append(l3)
        monthly = defaultdict(list)
        for week, children in sorted(weekly.items(), key=lambda item: item[1][0]['start']):
            iso_year, number = (int(x) for x in week.split('/')[1].split('-W'))
            l4 = self._summarize('L4', week, children, list(done['L4']),
                                 year=iso_year, week_number=number,
                                 week_start=children[0]['start'], week_end=children[-1]['end'])
            done['L4'].append(l4)
            monthly[week[:7]].append(l4)
        for month, children in sorted(monthly.items()):
            l5 = self._summarize('L5', month, children, list(done['L5']),
                                 year=int(month[:4]), month=int(month[5:]),
                                 month_start=children[0]['start'], month_end=children[-1]['end'])
            done['L5'].append(l5)
        self.data.set_metadata('finalized_sessions', [s.id for s in sessions])
        self.pending = False

    # ---------------------------------------------------------------- recall

    def _recall_adapter(self, temperature):
        from llm.base_llm import ModelConfig
        from llm.gemini_adapter import GeminiAdapter
        if self._recall_llm is None:
            self._recall_llm = make_llm(self.model, self.usage)
        adapter = GeminiAdapter(ModelConfig(model_name=self.model, temperature=temperature))
        adapter._llm = self._recall_llm
        return adapter

    def _native_planner(self):
        with self._recall_lock:
            if self._planner is None:
                from timem.workflows.retrieval_nodes.llm_retrieval_planner import LLMRetrievalPlanner
                planner = LLMRetrievalPlanner(llm_provider='gemini')
                planner.llm = self._recall_adapter(0.0)
                self._planner = planner
        return self._planner

    def _native_refiner(self):
        with self._recall_lock:
            if self._refiner is None:
                from timem.workflows.retrieval_nodes.memory_refiner import MemoryRefiner
                refiner = MemoryRefiner(llm_provider='gemini')
                refiner.llm = self._recall_adapter(0.0)
                self._refiner = refiner
        return self._refiner

    @staticmethod
    def _recall_config():
        from timem.utils.retrieval_config_manager import get_retrieval_config_manager
        return get_retrieval_config_manager().get_config()

    def _memory(self, node, **extra):
        layer, key = node['layer'], node['key']
        session = key.split('#')[0] if layer == 'L1' else (key if layer == 'L2' else '')
        span = node['start'] if node['start'] == node['end'] else f"{node['start']} to {node['end']}"
        return {'id': f'{layer}/{key}', 'level': layer, 'key': key, 'title': f'{layer} {span}',
                'content': node['content'], 'created_at': node['start'], 'start': node['start'],
                'end': node['end'], 'session_id': session, **extra}

    def _rank_l1(self, question, keywords, strategy, cfg):
        """Upstream perform_l1_retrieval: coarse dense + BM25 merge, fused top-k, then time order."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        coarse = strategy['layer_limits'].get('L1', 40)
        final = strategy['final_limits'].get('L1', 20)
        semantic_weight = cfg['retrieval']['semantic']['weight']
        keyword_weight = cfg['retrieval']['keyword']['weight']
        l1 = [n for (layer, _), n in sorted(self.nodes.items()) if layer == 'L1']
        hits = self._qdrant.query_points(
            self._collection, query=self._embeddings().embed_query(question), limit=coarse,
            query_filter=Filter(must=[FieldCondition(key='layer', match=MatchValue(value='L1'))]),
            with_payload=True).points
        semantic = {p.payload['key']: p.score for p in hits}
        bm25 = dict(zip((n['key'] for n in l1), _bm25([n['content'] for n in l1], ' '.join(keywords)))) \
            if keywords else {}
        max_bm25 = max(bm25.values(), default=0) or 1.0 if bm25 else 1.0
        max_sem = max(semantic.values(), default=0) or 1.0 if semantic else 1.0
        pool = []
        for node in l1:
            if node['key'] not in bm25 and node['key'] not in semantic:
                continue
            b, s = bm25.get(node['key'], 0.0), semantic.get(node['key'], 0.0)
            fused = (b / max_bm25 if max_bm25 > 0 else 0) * keyword_weight + \
                    (s / max_sem if max_sem > 0 else 0) * semantic_weight
            pool.append(self._memory(node, bm25_score=b, semantic_score=s, fused_score=fused))
        pool.sort(key=lambda m: m['fused_score'], reverse=True)
        return sorted(pool[:final], key=lambda m: (m['created_at'], m['key']))

    def _parents(self, l1_results, levels, counts):
        """Upstream bottom-up chain: walk the ranked L1s, collecting each level's ancestor once."""
        collected = {level: [] for level in levels}
        for rank, memory in enumerate(l1_results, 1):
            if all(len(collected[level]) >= counts.get(level, 0) for level in levels):
                break
            day = memory['start']
            chain = {'L2': memory['session_id'], 'L3': day, 'L4': _week_key(day), 'L5': day[:7]}
            for level in levels:
                node = self.nodes.get((level, chain[level]))
                if node is None or len(collected[level]) >= counts.get(level, 0):
                    continue
                if any(m['key'] == node['key'] for m in collected[level]):
                    continue
                collected[level].append(self._memory(node, chain_origin_l1_rank=rank))
        return collected

    @staticmethod
    def _order(memories, strategy_name, layers):
        if strategy_name == 'complex':
            for m in memories:
                m['weighted_score'] = m.get('fused_score', 0.5) * IMPORTANCE.get(m['level'], 0.3)
            return sorted(memories, key=lambda m: m['weighted_score'], reverse=True)
        order = [layer for layer in LAYERS if strategy_name == 'hybrid' or layer in layers]
        return [m for layer in order
                for m in sorted((m for m in memories if m['level'] == layer), key=lambda m: m['created_at'])]

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        meta = {'source_cutoff': self.cutoff, 'retrieval_profile': 'timem_complexity_aware'}
        if not any(layer == 'L1' for layer, _ in self.nodes):
            return answer_from_context(self.llm, q, '(no memories)', {**meta, 'hits': 0})
        cfg = self._recall_config()
        # Answers run on several threads; recall clients are synchronous, so each call gets its own loop.
        complexity, keywords = asyncio.run(
            self._native_planner().analyze_query_complexity(q.question, max_retries=3))
        strategy_name = STRATEGIES[complexity]
        strategy = cfg['retrieval_strategies'][strategy_name]
        layers = strategy['layers']
        l1_results = self._rank_l1(q.question, keywords, strategy, cfg)
        targets = [layer for layer in layers if layer != 'L1']
        parents = self._parents(l1_results, targets, strategy['final_limits'])
        candidates = l1_results + [m for layer in targets for m in parents[layer]]
        candidates = self._order(candidates, strategy_name, layers)
        state = asyncio.run(self._native_refiner().run(
            {'question': prompt_for(q), 'ranked_results': candidates, 'query_complexity': complexity}))
        kept = state.get('ranked_results', candidates)
        lines = [f"[{m['level']}] ({m['start']} to {m['end']}) {m['content']}" for m in kept]
        return answer_from_context(self.llm, q, '\n\n'.join(lines) or '(no memories)', {
            **meta, 'hits': len(kept), 'complexity': complexity, 'strategy': strategy_name,
            'keywords': keywords, 'candidates': len(candidates),
            'refiner_failed': bool(state.get('memory_refiner_failed')),
            'layers': [m['level'] for m in kept]})

    def close(self):
        try:
            for client in (self.llm, self._recall_llm):
                close_model = getattr(client, 'aclose', None)
                if close_model is not None:
                    self._generate(close_model())
            if self._loop is not None:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
        finally:
            if self._loop is not None:
                self._loop.close()
            self._qdrant.close()
