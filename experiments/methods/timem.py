"""TiMem (Findings of ACL 2026): 5-level Temporal Memory Tree with semantic-guided consolidation.

Uses the timem-ai SDK against a self-hosted or cloud TiMem server. Like Mem0, TiMem's
extraction and consolidation calls happen inside the service and are not counted in
usage.json. Set TIMEM_API_KEY and optionally TIMEM_BASE_URL (default: http://localhost:8001).
"""

import os

from .. import config
from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session, prompt_for
from ._common import answer_from_context, make_llm


class TiMem(MemoryMethod):
    name = "timem"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from timem import Memory
        self.user_id = instance.id
        self.character_id = "assistant"
        self.memory = Memory(
            api_key=os.getenv("TIMEM_API_KEY", "dev"),
            base_url=os.getenv("TIMEM_BASE_URL", "http://localhost:8001"),
        )
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        messages = []
        for t in session.turns:
            role = t.speaker if t.speaker in ("user", "assistant") else "user"
            name = "" if t.speaker in ("user", "assistant") else f"{t.speaker}: "
            messages.append({"role": role, "content": f"[{session.date}] {name}{t.text}"})
        result = self.memory.add(
            messages,
            user_id=self.user_id,
            character_id=self.character_id,
            session_id=session.id,
            metadata={"date": session.date},
        )
        return {"memories_created": result.get("total", 0)}

    def answer(self, q: Question) -> Answer:
        result = self.memory.search(
            query=prompt_for(q),
            user_id=self.user_id,
            limit=config.TOP_K,
        )
        memories = result.get("memories", [])
        lines = []
        for m in memories:
            layer = m.get("layer", "?")
            text = m.get("memory", "")
            lines.append(f"- [L{layer}] {text}")
        context = "\n".join(lines) or "(no memories)"
        return answer_from_context(self.llm, q, context, {"hits": len(memories)})
