"""
LifelongAgent — the main interface.

Usage:
    agent = LifelongAgent(base_dir="memory_v2/data")

    # Ingest a day's conversations (extracts memories + stores raw convs)
    agent.ingest(date="2026-03-01", conversations=[...])

    # After ingesting all data, build summaries
    agent.build_summaries()

    # Chat with full memory context
    response = agent.chat("What coffee do I usually drink in the morning?")
"""

import os
import json
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .store import MemoryStore
from .extractor import MemoryExtractor
from .summarizer import Summarizer
from .context_builder import ContextBuilder

import dotenv
dotenv.load_dotenv()

SYSTEM_PROMPT = """\
You are a personalized lifelong AI assistant. You have access to a rich memory \
of the user's past conversations, preferences, facts, goals, and events.

Use the provided memory context to give specific, accurate, and personalised responses. \
When memory is available, reference it naturally. \
When something is uncertain, say so rather than guessing.\
"""


class LifelongAgent:
    def __init__(self, base_dir: str, chat_model: str = "gpt-4o", extract_model: str = "gpt-4o-mini"):
        self.store   = MemoryStore(base_dir)
        self.extractor = MemoryExtractor(self.store, model=extract_model)
        self.summarizer = Summarizer(self.store, model=extract_model)
        self.ctx_builder = ContextBuilder(self.store, model=extract_model)
        self.chat_llm = ChatOpenAI(model=chat_model, temperature=0.7)

    # ── Ingestion ───────────────────────────────────────────────────

    def ingest(self, date: str, conversations: List[Dict]) -> List[Dict]:
        """
        Ingest one day's conversations.
        Extracts memories, deduplicates, stores to SQL + raw FAISS.

        conversations: [{time_of_day, turns: [{speaker, text}]}]
        Returns list of newly stored memories.
        """
        return self.extractor.extract_and_store(date, conversations)

    def ingest_from_json(self, filepath: str):
        """
        Ingest from a JSON file in the same format as learning_conversations.json:
        {"YYYY-MM-DD": {"day": "...", "interactions": [...]}, ...}
        """
        with open(filepath) as f:
            data = json.load(f)

        total = 0
        for date in sorted(data.keys()):
            info = data[date]
            convs = info.get("conversations") or info.get("interactions", [])
            if not convs:
                continue
            print(f"Ingesting {date}...")
            new = self.ingest(date, convs)
            total += len(new)

        print(f"\nIngestion complete. {total} new memories stored across {len(data)} days.")

    # ── Summarization ───────────────────────────────────────────────

    def build_summaries(self, force: bool = False):
        """Build the full weekly → monthly → yearly → lifetime summary hierarchy."""
        print("Building summary hierarchy...")
        self.summarizer.run(force=force)
        print("Summaries complete.")

    # ── Chat ────────────────────────────────────────────────────────

    def chat(self, query: str, n_recent: int = 6, n_relevant: int = 5) -> str:
        """
        Generate a response with full memory context.
        Context window: system_prompt + global_summary + relevant_memory + recent_convs
        """
        context = self.ctx_builder.build(query, n_recent=n_recent, n_relevant=n_relevant)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\n" + context),
            ("human",  "{query}"),
        ])

        chain = prompt | self.chat_llm
        response = chain.invoke({"query": query})
        return response.content.strip()

    # ── Inspection helpers ──────────────────────────────────────────

    def show_memories(self, type: Optional[str] = None, subject: Optional[str] = None, limit: int = 20):
        rows = self.store.query_memories(type=type, subject=subject, limit=limit)
        for r in rows:
            print(f"[{r['date']}] ({r['type']}/{r['subject']}) {r['content']}")

    def show_context(self, query: str):
        """Print the context that would be injected for a given query."""
        ctx = self.ctx_builder.build(query)
        print(ctx)

    def show_lifetime_summary(self):
        s = self.store.get_lifetime_summary()
        print(s or "No lifetime summary yet — run build_summaries() first.")
