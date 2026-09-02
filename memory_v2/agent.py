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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

from .store import MemoryStore
from .extractor import MemoryExtractor
from .summarizer import Summarizer
from .context_builder import ContextBuilder

import dotenv
dotenv.load_dotenv()

class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for g in generation:
                if hasattr(g, 'generation_info') and g.generation_info:
                    usage = g.generation_info.get('usage', {})
                    # OpenAI style
                    self.prompt_tokens += usage.get('prompt_tokens', 0)
                    self.completion_tokens += usage.get('completion_tokens', 0)
                    self.total_tokens += usage.get('total_tokens', 0)
                
                # Gemini style usage metadata
                if hasattr(g.message, 'usage_metadata') and g.message.usage_metadata:
                    um = g.message.usage_metadata
                    self.prompt_tokens += um.get('input_tokens', 0)
                    self.completion_tokens += um.get('output_tokens', 0)
                    self.total_tokens += um.get('total_tokens', 0)

    def get_report(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

SYSTEM_PROMPT = """\
You are a personalized lifelong AI assistant. You have access to a rich memory of the user's past conversations, preferences, facts, goals, and events.

Use the provided memory context to give specific, accurate, and personalised responses.
Be CONCISE: unless the question asks to explain the reasoning, answer in as few words as possible. Give the direct answer only — no preamble, no filler phrases, no "Based on our conversations..." or "I remember...".

If the question asks to "Explain the reasoning" or "Explain the reasoning.", you MUST format your response EXACTLY like this:
[Answer]. Reasoning: [Step-by-step logic and calculation]

Instructions for reasoning:
1. First, in your thoughts, calculate the exact math, sequences, dates, and transitions step-by-step.
2. Formulate the core, final, correct answer.
3. State the core answer, followed by a period.
4. Write the literal word "Reasoning: " followed by your calculated step-by-step logic.
Do not use lists/bullet points in the reasoning if not absolutely necessary. Keep it as a concise paragraph.

If the information is not in memory, say only: "I don't know."
"""

CANONICAL_SUBJECT_PROMPT = """\
You are a domain taxonomy expert.
I have a group of similar preference subjects extracted from a user's conversations.
I need to pick ONE canonical name for this group.

Group of subjects:
{subjects}

Rules:
1. Identify if these subjects are specific variations of a single, high-level recurring category.
2. Pick the most stable and general name for that category.
3. If the subjects are activities, group them under a high-level action category (e.g. "fitness", "hobbies").
4. If the subjects are items or preferences, group them under their core domain.
5. Avoid redundant words and keep it to a clean noun or noun-phrase.

Return ONLY the canonical name as a plain string.
"""


class LifelongAgent:
    def __init__(self, base_dir: str, chat_model: str = "gpt-5.5", extract_model: str = "gpt-5.5", callbacks=None):
        self.store   = MemoryStore(base_dir)
        
        def get_llm(model_name, temperature=0):
            if "gemini" in model_name.lower():
                return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, callbacks=callbacks)
            else:
                return ChatOpenAI(model=model_name, temperature=temperature, callbacks=callbacks)

        self.extractor = MemoryExtractor(self.store, llm=get_llm(extract_model, 0))
        self.summarizer = Summarizer(self.store, llm=get_llm(extract_model, 0.2))
        self.ctx_builder = ContextBuilder(self.store, llm=get_llm(extract_model, 0))
        self.chat_llm = get_llm(chat_model, 0.7)

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

    def consolidate_memories(self):
        """Merge fragmented subjects in the memory store using LLM-based reasoning."""
        print(f"Consolidating domains using LLM reasoning...")
        
        # 1. Get all subjects and sample values
        subjects = self.store.get_all_subjects()
        if not subjects:
            return

        subject_summary = []
        for subj in subjects:
            mems = self.store.query_memories(subject=subj, limit=5)
            vals = list({m["content"] for m in mems})
            subject_summary.append(f"- {subj}: {', '.join(vals[:3])}")

        # 2. Ask LLM to group them
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a taxonomy expert. Look at the following list of preference subjects "
             "and their sample values. Some subjects are duplicates or near-duplicates.\n\n"
             "Identify which subjects should be merged into a single canonical domain. Guidelines:\n"
             "- Merge daily routines, activities, and rest/workout status into a single canonical subject.\n"
             "- Merge apparel, garments, and clothing categories into a single canonical subject.\n"
             "- Merge drink, coffee, and beverage categories into a single canonical subject.\n"
             "Return ONLY a JSON mapping from the old subject name to the new canonical name.\n"
             "Include ALL subjects in the mapping, even if they stay the same.\n\n"
             "Example output: {{\"jazz\": \"music\", \"blues\": \"music\", \"hiking\": \"leisure\"}}"),
            ("human", "SUBJECTS:\n" + "\n".join(subject_summary)),
        ])
        
        try:
            from langchain_core.output_parsers import JsonOutputParser
            chain = prompt | self.chat_llm | JsonOutputParser()
            mapping = chain.invoke({})
            
            # 3. Apply to DB
            changes_made = False
            with self.store._conn() as conn:
                for old, new in mapping.items():
                    old_clean = old.lower().strip()
                    new_clean = new.lower().strip()
                    if old_clean != new_clean:
                        changes_made = True
                        print(f"  [Consolidation] Merging '{old_clean}' -> '{new_clean}'")
                        conn.execute(
                            "UPDATE memories SET subject = ? WHERE subject = ?",
                            (new_clean, old_clean)
                        )
            
            if not changes_made:
                print("  [Consolidation] No merges needed. Subjects are already canonical.")
            
            final_subjects = self.store.get_all_subjects()
            print(f"  [Taxonomy] Final subjects: {', '.join(final_subjects)}")

        except Exception as e:
            print(f"  Consolidation error: {e}")

    # ── Summarization ───────────────────────────────────────────────

    def build_summaries(self, force: bool = False):
        """Build the full weekly → monthly → yearly → lifetime summary hierarchy."""
        print("Consolidating memory subjects...")
        self.consolidate_memories()
        
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
        safe_context = context.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\n" + safe_context),
            ("human",  "{query}"),
        ])

        chain = prompt | self.chat_llm
        response = chain.invoke({"query": query})
        
        if isinstance(response.content, list):
            # Some models like Gemini might return content blocks
            return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip()
        return str(response.content).strip()

    # ── Inspection helpers ──────────────────────────────────────────

    def show_memories(self, subject: Optional[str] = None, limit: int = 20):
        rows = self.store.query_memories(subject=subject, limit=limit)
        for r in rows:
            print(f"[{r['date']}] ({r['subject']}) {r['content']}")

    def show_context(self, query: str):
        """Print the context that would be injected for a given query."""
        ctx = self.ctx_builder.build(query)
        print(ctx)

    def show_lifetime_summary(self):
        s = self.store.get_lifetime_summary()
        print(s or "No lifetime summary yet — run build_summaries() first.")
