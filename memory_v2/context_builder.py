"""
Context builder.

Assembles the final context window:
  system_prompt
  + [GLOBAL MEMORY]      — per-speaker lifetime summaries
  + [RELEVANT MEMORY]    — SQL (filtered by speaker) or FAISS TopK
  + [RECENT CONVERSATIONS] — last K raw conversation chunks

Query classifier now also detects the speaker name so retrieval
can be scoped to the right person.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .store import MemoryStore


CLASSIFY_SYSTEM = """\
Classify the user query and extract key metadata.

Return ONLY JSON:
{{
  "category": "factual" | "preference" | "goal" | "episodic" | "general",
  "subjects":  ["entity1", ...],
  "speaker":   "name of the person the question is about, or null if unclear",
  "target_date": "YYYY-MM-DD or null"
}}

category meanings:
  factual    — stable personal fact (name, job, location, relationship)
  preference — what someone likes/prefers about something
  goal       — intentions, plans, aspirations
  episodic   — specific past event or what happened on a date
  general    — anything else or mixed

speaker: use the actual name if a person is named (e.g. "caroline", "melanie").
         Use "user" for single-person systems or if unclear.

target_date: If a specific date is mentioned, format as YYYY-MM-DD (assume year is 2026 if missing).
             If only a month is mentioned (e.g., "March"), use the 15th of that month (e.g., "2026-03-15").
             If no time frame is mentioned, return null.
"""


class ContextBuilder:
    def __init__(self, store: MemoryStore, model: str = "gpt-4o-mini", llm=None):
        self.store = store
        if llm is not None:
            self.llm = llm
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, temperature=0)

    def build(
        self,
        query: str,
        n_recent: int = 3,
        n_relevant: int = 5,
        n_conv_search: int = 10,
    ) -> str:
        classification = self._classify(query)
        category = classification.get("category", "general")
        subjects = classification.get("subjects", [])
        speaker  = classification.get("speaker") or None
        target_date = classification.get("target_date") or None

        if speaker:
            speaker = speaker.lower().strip()
            known = self.store.get_all_speakers()
            if speaker not in known:
                speaker = None

        global_summary  = self._global_summary(speaker)
        relevant_memory = self._relevant_memory(category, subjects, query,
                                                n_relevant, speaker)
        # Always do semantic search over raw conversation chunks
        semantic_convs  = self._faiss_conversations(query, k=n_conv_search)
        
        # Recent chunks by date or target window
        exclude_text = relevant_memory + semantic_convs
        if target_date:
            recent_convs = self._windowed_conversations(target_date, days_window=5, exclude=exclude_text)
        else:
            recent_convs = self._recent_conversations(n_recent, exclude=exclude_text)

        return self._format(global_summary, relevant_memory,
                            semantic_convs, recent_convs, speaker)

    # ── Classification ──────────────────────────────────────────────

    def _classify(self, query: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFY_SYSTEM),
            ("human",  "{query}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            return chain.invoke({"query": query})
        except Exception:
            return {"category": "general", "subjects": [], "speaker": None}

    # ── Memory retrieval ────────────────────────────────────────────

    def _global_summary(self, speaker: Optional[str] = None) -> str:
        """
        Return per-speaker lifetime summary when speaker is known,
        otherwise return all speakers' lifetime summaries combined.
        """
        known_speakers = self.store.get_all_speakers()

        if speaker and speaker in known_speakers:
            text = self.store.get_lifetime_summary(speaker)
            return self._strip_prefix(text) if text else ""

        # Return all speakers' lifetime summaries
        parts = []
        for sp in known_speakers:
            text = self.store.get_lifetime_summary(sp)
            if text:
                content = self._strip_prefix(text)
                parts.append(f"[{sp.capitalize()}]\n{content}")
        return "\n\n".join(parts)

    def _relevant_memory(
        self,
        category: str,
        subjects: List[str],
        query: str,
        k: int,
        speaker: Optional[str] = None,
    ) -> str:
        if category == "preference":
            return self._sql_preferences(subjects, k, speaker)
        else:
            # factual, goal, episodic, general: search summaries only
            # (facts are never stored in SQL — they come from FAISS conv search)
            summary_hits = self.store.search_summaries(query, k=k)
            if speaker:
                summary_hits = [
                    d for d in summary_hits
                    if d.metadata.get("speaker", "").lower() == speaker
                    or not d.metadata.get("speaker")
                ]
            return "\n\n---\n\n".join(d.page_content for d in summary_hits[:k]) if summary_hits else ""

    def _sql_preferences(self, subjects: List[str], k: int,
                          speaker: Optional[str] = None) -> str:
        lines = []
        for subj in subjects or [None]:
            rows = self.store.query_memories(
                subject=subj,
                speaker=speaker,
                limit=k * 3,
            )
            for r in rows[:k]:
                spk  = f"[{r['speaker']}] " if r.get("speaker") else ""
                date = r["date"] or ""
                lines.append(f"{spk}[{date}] {r['content']}")
        return "\n".join(lines) if lines else ""

    def _sql_typed(self, category: str, subjects: List[str], k: int,
                   speaker: Optional[str] = None) -> str:
        type_map = {"factual": "fact", "goal": "goal"}
        mem_type = type_map.get(category, category)
        lines = []
        for subj in (subjects or [None]):
            rows = self.store.query_memories(
                type=mem_type, subject=subj, speaker=speaker, limit=k,
            )
            for r in rows:
                spk  = f"[{r['speaker']}] " if r.get("speaker") else ""
                lines.append(f"{spk}[{r['date'] or ''}] {r['content']}")
        if not lines:
            rows = self.store.query_memories(
                subject=subjects[0] if subjects else None,
                speaker=speaker, limit=k,
            )
            lines = [
                f"[{r.get('speaker','')}] [{r['date'] or ''}] {r['content']}"
                for r in rows
            ]
        return "\n".join(lines)

    def _faiss_conversations(self, query: str, k: int) -> str:
        docs = self.store.search_conversations(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""

    def _recent_conversations(self, k: int, exclude: str = "") -> str:
        docs = self.store.get_recent_conversations(k=k)
        # Skip any doc whose content already appears in the relevant memory block
        unique = [d for d in docs if d.page_content[:100] not in exclude]
        return "\n\n---\n\n".join(d.page_content for d in unique) if unique else ""

    def _windowed_conversations(self, target_date: str, days_window: int = 5, exclude: str = "") -> str:
        from datetime import datetime, timedelta
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return ""
        
        start_date = (dt - timedelta(days=days_window)).strftime("%Y-%m-%d")
        end_date = (dt + timedelta(days=days_window)).strftime("%Y-%m-%d")
        
        docs = self.store.get_conversations_by_date_range(start_date, end_date)
        # Skip any doc whose content already appears in the exclude block
        unique = [d for d in docs if d.page_content[:100] not in exclude]
        return "\n\n---\n\n".join(d.page_content for d in unique) if unique else ""

    # ── Formatter ───────────────────────────────────────────────────

    def _format(
        self,
        global_summary: str,
        relevant_memory: str,
        semantic_convs: str,
        recent_convs: str,
        speaker: Optional[str] = None,
    ) -> str:
        parts = []
        label = f"GLOBAL MEMORY ({speaker.capitalize()})" if speaker else "GLOBAL MEMORY"

        if global_summary:
            parts.append(f"[{label}]\n{global_summary}")
        if relevant_memory:
            parts.append(f"[RELEVANT MEMORY]\n{relevant_memory}")
        if semantic_convs:
            parts.append(f"[RELEVANT CONVERSATIONS]\n{semantic_convs}")
        if recent_convs:
            parts.append(f"[RECENT CONVERSATIONS]\n{recent_convs}")

        return "\n\n" + "\n\n".join(parts) if parts else ""

    @staticmethod
    def _strip_prefix(text: str) -> str:
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else text
