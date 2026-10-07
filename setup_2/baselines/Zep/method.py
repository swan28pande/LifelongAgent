"""Zep, self-hosted: Graphiti (graphiti-core 0.30.2), Zep's open-source temporal knowledge graph.

Zep Cloud is not available to this project, so the graph engine Zep is built on runs locally on an
embedded Kuzu database inside the user's store, which the monthly runner backs up and restores as a
unit. Each session becomes one message episode dated at the session's date; Graphiti extracts
entities and facts, deduplicates them against the graph, and sets each fact's validity interval,
invalidating facts that a later episode contradicts.

Retrieval follows the Zep LoCoMo evaluation used for the paper's LoCoMo row: 20 facts (hybrid BM25
and cosine search, cross-encoder reranking) and 20 entities (hybrid search, RRF), formatted in
Zep's FACTS/ENTITIES context. Graphiti's own prompts build the graph with the run's Gemini model
for every call; the shared Nomic embedder, a local BGE cross-encoder and the shared answer prompt
complete the setup.
"""

import asyncio
import datetime as dt
import os
import threading
import time

os.environ.setdefault('EMBEDDING_DIM', '768')  # Read by graphiti_core at import time.
os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session, prompt_for
from .._common import answer_from_context, embeddings, make_llm
from ..store import Sessions

MAX_OUTPUT_TOKENS = 32768   # Gemini counts thinking tokens against the limit.
SEARCH_LIMIT = 20

CONTEXT = """FACTS and ENTITIES represent relevant context from the conversation.
For temporal questions, use the event times on facts.

<FACTS>
{facts}
</FACTS>

<ENTITIES>
{entities}
</ENTITIES>"""


def _graphiti_classes():
    from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.gemini_client import GeminiClient

    class StrictGemini(GeminiClient):
        """Same model for every call, one output limit, and no partial-JSON salvage.

        Graphiti sends some prompts to a small default model (gemini-2.5-flash-lite), caps edge
        extraction at 16K output tokens, and on truncated output keeps whatever JSON parses.
        Here every call uses the run's model, and an unusable reply raises so Graphiti retries.
        """

        def _get_model_for_size(self, model_size):
            return self.model

        def _resolve_max_tokens(self, requested_max_tokens, model):
            return MAX_OUTPUT_TOKENS

        def salvage_json(self, raw_output):
            return None

    class NomicEmbedder(EmbedderClient):
        """The shared Nomic model; Graphiti embeds names, facts and queries with one call."""

        async def create(self, input_data):
            text = input_data if isinstance(input_data, str) else str(input_data)
            return (await self.create_batch([text]))[0]

        async def create_batch(self, input_data_list):
            return await asyncio.to_thread(embeddings().embed_documents, list(input_data_list))

    return StrictGemini, NomicEmbedder, BGERerankerClient


class _Metered:
    """Counts every Gemini call Graphiti makes (thinking included) into the run's usage."""

    def __init__(self, models, usage):
        self._models, self._usage = models, usage

    async def generate_content(self, **kwargs):
        response = await self._models.generate_content(**kwargs)
        meta = getattr(response, 'usage_metadata', None)
        with self._usage.lock:
            self._usage.calls += 1
            if meta is not None:
                self._usage.input_tokens += meta.prompt_token_count or 0
                self._usage.output_tokens += (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        return response

    def __getattr__(self, name):
        return getattr(self._models, name)


class _MeteredClient:
    """A genai.Client stand-in whose client.aio.models is metered (the SDK property is read-only)."""

    def __init__(self, client, usage):
        self._client = client
        self.aio = type('Aio', (), {'models': _Metered(client.aio.models, usage)})()

    def __getattr__(self, name):
        return getattr(self._client, name)


class Zep(MemoryMethod):
    name = 'zep'
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from google import genai
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.llm_client.config import LLMConfig
        from memory_v3.agent import _vertex_project

        StrictGemini, NomicEmbedder, BGEReranker = _graphiti_classes()
        self.data = Sessions(store_dir)
        self.group = instance.id
        self.user_name = instance.user or 'user'
        client = _MeteredClient(genai.Client(vertexai=True, project=_vertex_project(),
                                             location=os.getenv('GOOGLE_CLOUD_LOCATION', 'global')), usage)
        llm = StrictGemini(LLMConfig(model=model, small_model=model, temperature=0.0,
                                     max_tokens=MAX_OUTPUT_TOKENS), client=client)

        # Graphiti is async; one loop thread owns the graph, and callers block on it.
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.driver = KuzuDriver(db=str(store_dir / 'graph.kuzu'))
        # One database per user: name it after the group so Graphiti keeps using this driver
        # (KuzuDriver never sets _database, which add_episode compares the group id against).
        self.driver._database = self.group
        self._full_text_indexes()
        self.graph = Graphiti(graph_driver=self.driver, llm_client=llm, embedder=NomicEmbedder(),
                              cross_encoder=BGEReranker())
        self._run(self.graph.build_indices_and_constraints())
        self.pending = False
        self.llm = make_llm(model, usage)

    def _full_text_indexes(self) -> None:
        """Create the BM25 indexes Graphiti searches; KuzuDriver's build_indices is a no-op, and a
        missing index only logs an error, leaving search silently vector-only."""
        import kuzu
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        conn = kuzu.Connection(self.driver.db)
        try:
            for statement in ('INSTALL fts;', 'LOAD EXTENSION fts;'):
                conn.execute(statement)
            existing = {row[1] for row in conn.execute('CALL SHOW_INDEXES() RETURN *;').get_all()}
            for query in get_fulltext_indices(GraphProvider.KUZU):
                name = query.split("'")[3]
                if name not in existing:
                    conn.execute(query)
        finally:
            conn.close()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def ingest(self, session: Session) -> dict:
        existing = self.data.get(session.id)
        if existing is not None:
            if existing != session:
                raise ValueError(f'Observed session changed: {session.id}')
            return {'added': False}
        # Speaker names, as in Zep's LoCoMo ingestion, so the user is one entity rather than 'user'.
        body = '\n'.join(f"{self.user_name if t.speaker == 'user' else t.speaker}: {t.text}" for t in session.turns)
        when = dt.datetime.fromisoformat(f'{session.date}T{session.time or "12:00"}').replace(tzinfo=dt.timezone.utc)
        for attempt in range(4):
            try:
                result = self._run(self.graph.add_episode(
                    name=f'session {session.id}', episode_body=body, reference_time=when,
                    source_description=f'conversation between {self.user_name} and their AI assistant',
                    group_id=self.group))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))
        # Recorded only after Graphiti stored the episode; a failed month is rolled back as a whole.
        self.data.add(session)
        self.pending = True
        return {'added': True, 'nodes': len(result.nodes), 'edges': len(result.edges),
                'invalidated': sum(1 for e in result.edges if e.invalid_at is not None)}

    def finalize(self) -> None:
        # Flush Kuzu's write-ahead log so the next month's backup copies a consistent database.
        self._run(self.driver.execute_query('CHECKPOINT;'))
        self.pending = False

    def _search(self, query: str):
        from graphiti_core.search.search_config import (
            EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, NodeReranker, NodeSearchConfig,
            NodeSearchMethod, SearchConfig)

        edges = SearchConfig(limit=SEARCH_LIMIT, edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder))
        nodes = SearchConfig(limit=SEARCH_LIMIT, node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf))
        e = self._run(self.graph.search_(query, config=edges, group_ids=[self.group]))
        n = self._run(self.graph.search_(query, config=nodes, group_ids=[self.group]))
        return e.edges, n.nodes

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        edges, nodes = self._search(prompt_for(q))

        def when(edge):
            start = edge.valid_at.strftime('%Y-%m-%dT%H:%M:%SZ') if edge.valid_at else 'unknown'
            end = f'; no longer true from {edge.invalid_at.strftime("%Y-%m-%d")}' if edge.invalid_at else ''
            return f'(event_time: {start}{end})'

        context = CONTEXT.format(facts='\n'.join(f'- {e.fact} {when(e)}' for e in edges),
                                 entities='\n'.join(f'- {n.name}: {n.summary}' for n in nodes))
        return answer_from_context(self.llm, q, context, {'edges': len(edges), 'nodes': len(nodes),
                                                         'invalid_edges': sum(1 for e in edges if e.invalid_at)})

    def close(self) -> None:
        # KuzuDriver.close() is a no-op; release the database so the store can be reopened or copied.
        try:
            self.driver.client.close()
            self.driver.db.close()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
