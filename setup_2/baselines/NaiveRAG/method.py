"""Control: fixed-size chunks of raw turns, dense top-k retrieval, one answer call."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session, prompt_for
from .._common import answer_from_context, embeddings, make_llm
from ..store import Sessions


class NaiveRAG(MemoryMethod):
    name = "naive_rag"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        self.data = Sessions(store_dir)
        saved = self.data.metadata('index', {'sessions': [], 'chunks': 0})
        self.indexed = set(saved['sessions'])
        sessions = self.data.all()
        if self.indexed - {s.id for s in sessions}:
            raise RuntimeError('Saved index references missing observations')
        self.pending = [s for s in sessions if s.id not in self.indexed]
        self.index = None
        if saved['chunks']:
            # Load only the trusted index written inside this run's store.
            self.index = FAISS.load_local(str(store_dir / 'faiss'), embeddings(),
                                          allow_dangerous_deserialization=True)
            if self.index.index.ntotal != saved['chunks']:
                raise RuntimeError('Saved FAISS index is incomplete')
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        added = self.data.add(session)
        if added:
            self.pending.append(session)
        return {'added': added}

    def finalize(self) -> None:
        if not self.pending:
            return
        docs = []
        for session in self.pending:
            for i in range(0, len(session.turns), config.CHUNK_TURNS):
                body = "\n".join(f"{t.speaker}: {t.text}" for t in session.turns[i:i + config.CHUNK_TURNS])
                docs.append(Document(page_content=f'Date: {session.date}\n{body}',
                                     metadata={'date': session.date, 'session_id': session.id}))
        if docs:
            if self.index is None:
                self.index = FAISS.from_documents(docs, embeddings())
            else:
                self.index.add_documents(docs)
            self.index.save_local(str(self.store_dir / 'faiss'))
        self.indexed.update(s.id for s in self.pending)
        self.data.set_metadata('index', {'sessions': sorted(self.indexed),
                                        'chunks': self.index.index.ntotal if self.index else 0})
        self.pending.clear()

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        docs = self.index.similarity_search(prompt_for(q), k=config.TOP_K) if self.index else []
        docs.sort(key=lambda d: d.metadata["date"])
        context = "\n\n".join(d.page_content for d in docs)
        return answer_from_context(self.llm, q, context, {"retrieved_dates": [d.metadata["date"] for d in docs]})
