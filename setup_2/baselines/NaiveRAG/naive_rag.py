"""
Naive RAG baseline.

Stores raw conversation chunks in a FAISS index (no extraction, no summaries).
For each query, retrieves top-k semantically similar chunks and answers from them.

This is the simplest possible memory baseline — no structured extraction,
no hierarchical summarization, just chunked conversations + semantic search.
"""

from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
CHUNK_SIZE  = 5   # turns per FAISS document


class NaiveRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
        )
        self._store: Optional[FAISS] = None

    def ingest_sessions(self, sessions: list):
        """
        sessions: list of (date_str, turns, session_key, speaker_a)
          where turns = [{"speaker": str, "text": str, ...}]
        """
        docs = []
        for date, turns, sk, speaker_a in sessions:
            for start in range(0, max(len(turns), 1), CHUNK_SIZE):
                batch = turns[start: start + CHUNK_SIZE]
                if not batch:
                    continue
                lines = [f"Date: {date}"]
                for t in batch:
                    if "text" not in t:
                        continue
                    role = "User" if t["speaker"] == speaker_a else "System"
                    lines.append(f"{role}: {t['text']}")
                docs.append(Document(
                    page_content="\n".join(lines),
                    metadata={"source_date": date, "session": sk},
                ))

        if docs:
            if self._store is None:
                self._store = FAISS.from_documents(docs, self.embeddings)
            else:
                self._store.add_documents(docs)

    def search(self, query: str, k: int = 5) -> str:
        """Return top-k chunks concatenated as context string."""
        if self._store is None:
            return ""
        hits = self._store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in hits)

    def reset(self):
        self._store = None
