"""Mem0 open-source (ECAI 2025): LLM fact extraction with ADD/UPDATE/DELETE/NOOP over a vector store.

Held equal to the other methods: Gemini on Vertex for extraction and answering, the shared
embedding model, and the shared answer prompt. The session date goes into each message and
into metadata, since mem0 OSS has no native per-message timestamp for extraction.
Mem0's own extraction calls are made inside the library and are not counted in usage.json.
"""

import os

from .. import config
from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session, prompt_for
from ._common import answer_from_context, make_llm


class Mem0OSS(MemoryMethod):
    name = "mem0"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from mem0 import Memory
        from memory_v3.agent import _vertex_project
        self.user_id = instance.id
        self.memory = Memory.from_config({
            "llm": {"provider": "gemini", "config": {
                # mem0's default max_tokens counts Gemini's thinking tokens and truncates the JSON.
                "model": model, "temperature": 0.0, "max_tokens": 32000, "vertexai": True,
                "project": _vertex_project(), "location": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            }},
            "embedder": {"provider": "huggingface", "config": {
                "model": config.EMBED_MODEL, "embedding_dims": config.EMBED_DIMS,
                "model_kwargs": {"trust_remote_code": True},
            }},
            "vector_store": {"provider": "faiss", "config": {
                "path": str(store_dir), "collection_name": "memories",
                "embedding_model_dims": config.EMBED_DIMS,
            }},
            "history_db_path": str(store_dir / "history.db"),
        })
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        messages = []
        for t in session.turns:
            role = t.speaker if t.speaker in ("user", "assistant") else "user"
            name = "" if t.speaker in ("user", "assistant") else f"{t.speaker}: "
            messages.append({"role": role, "content": f"[{session.date}] {name}{t.text}"})
        result = self.memory.add(messages, user_id=self.user_id, metadata={"date": session.date})
        return {"added": len(result.get("results", []))}

    def answer(self, q: Question) -> Answer:
        hits = self.memory.search(prompt_for(q), filters={"user_id": self.user_id}, top_k=config.TOP_K)
        hits = hits.get("results", []) if isinstance(hits, dict) else hits
        lines = [f"- [{(h.get('metadata') or {}).get('date', '?')}] {h['memory']}" for h in hits]
        return answer_from_context(self.llm, q, "\n".join(lines) or "(no memories)", {"hits": len(hits)})
