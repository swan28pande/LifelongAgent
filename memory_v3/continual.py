"""
Continual knowledge distillation with forward and backward transfer.

The hierarchical summariser (weekly → monthly → yearly → lifetime) produces
batch snapshots. This module maintains a **living knowledge document** per
speaker that is updated after every ingestion, not rebuilt from scratch.

Forward transfer  — the current knowledge state is injected into the
                    extraction prompt so new memories benefit from what
                    the system already knows (entity consistency, pattern
                    awareness, fact continuity).

Backward transfer — after new memories are stored, the knowledge document
                    is updated: new facts are added, changed facts record
                    the transition with dates, new patterns are registered,
                    and stale entries are marked.

The knowledge document is stored in the summary FAISS index with identifier
``knowledge:{speaker}`` so the retrieval agent can search it alongside
the hierarchical summaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .store import MemoryStore
from .prompts import KNOWLEDGE_INIT_SYSTEM, KNOWLEDGE_UPDATE_SYSTEM


class KnowledgeState:
    """Per-speaker distilled knowledge document, updated continually."""

    def __init__(self, store: MemoryStore, llm, speaker: str):
        self.store = store
        self.llm = llm
        self.speaker = speaker
        self._identifier = f"knowledge:{speaker}"

    # ── Read ───────────────────────────────────────────────────────

    def load(self) -> str:
        raw = self.store.get_summary(self._identifier)
        if not raw:
            return ""
        parts = raw.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else raw

    def exists(self) -> bool:
        return bool(self.load())

    # ── Forward transfer ───────────────────────────────────────────

    def forward_context(self) -> str:
        """Return knowledge state formatted for injection into extraction."""
        doc = self.load()
        if not doc:
            return ""
        return (
            f"CURRENT KNOWLEDGE STATE for {self.speaker} "
            f"(use this to maintain consistency):\n{doc}"
        )

    # ── Backward transfer ──────────────────────────────────────────

    def update(self, new_memories: List[Dict], date: str) -> str:
        """
        Update the knowledge document with newly extracted memories.

        If no document exists yet, initialise one. Otherwise, ask the LLM
        to integrate the new information — adding new facts, recording
        transitions for changed facts, and registering new patterns.

        Returns the updated document text.
        """
        current = self.load()

        mem_block = self._format_memories(new_memories)
        if not mem_block:
            return current

        if not current:
            updated = self._init(mem_block, date)
        else:
            updated = self._update(current, mem_block, date)

        self.store.save_summary(
            self._identifier,
            f"Knowledge State: {self.speaker}",
            updated,
            {"speaker": self.speaker, "last_updated": date},
        )
        return updated

    # ── Internals ──────────────────────────────────────────────────

    def _init(self, memories_block: str, date: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_INIT_SYSTEM),
            ("human",
             f"Speaker: {self.speaker}\n"
             f"Date: {date}\n\n"
             f"MEMORIES:\n{_esc(memories_block)}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"speaker": self.speaker})
            return result.get("document", "")
        except Exception as e:
            print(f"  Knowledge init error: {e}")
            return ""

    def _update(self, current: str, memories_block: str, date: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_UPDATE_SYSTEM),
            ("human",
             f"Speaker: {self.speaker}\n"
             f"Date: {date}\n\n"
             f"CURRENT KNOWLEDGE DOCUMENT:\n{_esc(current)}\n\n"
             f"NEW MEMORIES:\n{_esc(memories_block)}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"speaker": self.speaker})
            return result.get("document", "")
        except Exception as e:
            print(f"  Knowledge update error: {e}")
            return current

    @staticmethod
    def _format_memories(memories: List[Dict]) -> str:
        if not memories:
            return ""
        lines = []
        for m in memories:
            speaker = m.get("speaker", "")
            mtype = m.get("type", "")
            lines.append(
                f"[{m.get('date', '?')}] ({mtype}) "
                f"{m['entity']}: {m['content']}"
                + (f" [{speaker}]" if speaker else "")
            )
        return "\n".join(lines)


def _esc(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")
