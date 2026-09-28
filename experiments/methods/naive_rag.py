"""Control: fixed-size chunks of raw turns, dense top-k retrieval, one answer call."""

from langchain_community.vectorstores import FAISS

from .. import config
from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session, prompt_for
from ._common import answer_from_context, embeddings, make_llm


class NaiveRAG(MemoryMethod):
    name = "naive_rag"

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        self.chunks: list[tuple[str, str]] = []   # (date, text)
        self.index = None
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        turns = session.turns
        for i in range(0, len(turns), config.CHUNK_TURNS):
            body = "\n".join(f"{t.speaker}: {t.text}" for t in turns[i:i + config.CHUNK_TURNS])
            self.chunks.append((session.date, f"Date: {session.date}\n{body}"))
        return {}

    def finalize(self) -> None:
        self.index = FAISS.from_texts([c for _, c in self.chunks], embeddings(),
                                      metadatas=[{"date": d} for d, _ in self.chunks])

    def answer(self, q: Question) -> Answer:
        docs = self.index.similarity_search(prompt_for(q), k=config.TOP_K)
        docs.sort(key=lambda d: d.metadata["date"])
        context = "\n\n".join(d.page_content for d in docs)
        return answer_from_context(self.llm, q, context, {"retrieved_dates": [d.metadata["date"] for d in docs]})
