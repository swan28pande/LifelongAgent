"""
Knowledge distillation from hierarchical summaries.

The summarizer builds temporal snapshots (weekly → monthly → yearly → lifetime).
The distiller reasons over those snapshots to produce focused knowledge
documents that track temporal evolution across four themes:

  relationships — social connections between people and how they evolve
  identity      — stable facts and how they changed over time
  patterns      — behavioral routines with start/end dates
  timeline      — chronological milestones and key transitions

Each document records transitions with dates and reasoning:

    "Priya and Rohan: close friends who run together (since Mar 2026).
     Relationship cooled after the project disagreement (Jul 2026)."

These sit at the top of the memory pyramid — derived from extensive
reasoning over summaries, not simple aggregation.

    raw memories (DB)
        → hierarchical summaries (weekly/monthly/yearly/lifetime)
            → distilled knowledge documents (reasoned, multi-theme)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .store import MemoryStore
from .prompts import DISTILL_SYSTEM, DISTILL_UPDATE_SYSTEM, THEME_DESCRIPTIONS


THEMES = ("relationships", "identity", "patterns", "timeline")


class KnowledgeDistiller:
    """Distills hierarchical summaries into themed knowledge documents."""

    def __init__(self, store: MemoryStore, llm):
        self.store = store
        self.llm = llm

    def distill(self, speaker: str) -> Dict[str, str]:
        """
        Create or update all knowledge documents for a speaker.

        Reads monthly, yearly, and lifetime summaries, then for each theme
        either creates a new document or updates the existing one.
        """
        summaries = self._gather_summaries(speaker)
        if not summaries:
            return {}

        results = {}
        for theme in THEMES:
            current = self._load(speaker, theme)
            if current:
                updated = self._update(speaker, theme, current, summaries)
            else:
                updated = self._create(speaker, theme, summaries)

            if updated:
                self._save(speaker, theme, updated)
                results[theme] = updated
                print(f"  ✓ Distilled: {theme}:{speaker}")

        return results

    def _gather_summaries(self, speaker: str) -> str:
        """Collect all available summaries for a speaker into one block."""
        sections = []

        lifetime = self._get(f"lifetime:{speaker}")
        if lifetime:
            sections.append(f"=== LIFETIME ===\n{lifetime}")

        for doc in self._all_docs():
            ident = doc.metadata.get("identifier", "")
            if ident.startswith("year:") and ident.endswith(f":{speaker}"):
                label = ident.split(":")[1]
                content = self._strip_header(doc.page_content)
                sections.append(f"=== YEAR {label} ===\n{content}")

        for doc in self._all_docs():
            ident = doc.metadata.get("identifier", "")
            if ident.startswith("month:") and ident.endswith(f":{speaker}"):
                label = ident.split(":")[1]
                content = self._strip_header(doc.page_content)
                sections.append(f"=== MONTH {label} ===\n{content}")

        return "\n\n".join(sections)

    def _create(self, speaker: str, theme: str, summaries: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DISTILL_SYSTEM),
            ("human",
             f"Speaker: {speaker}\n"
             f"Theme: {theme}\n\n"
             f"ALL SUMMARIES:\n{_esc(summaries)}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "speaker": speaker,
                "theme": theme,
                "theme_description": THEME_DESCRIPTIONS[theme],
            })
            return result.get("document", "")
        except Exception as e:
            print(f"  ✗ Distill create error ({theme}): {e}")
            return ""

    def _update(self, speaker: str, theme: str, current: str,
                summaries: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DISTILL_UPDATE_SYSTEM),
            ("human",
             f"Speaker: {speaker}\n"
             f"Theme: {theme}\n\n"
             f"CURRENT DOCUMENT:\n{_esc(current)}\n\n"
             f"ALL SUMMARIES:\n{_esc(summaries)}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "speaker": speaker,
                "theme": theme,
                "theme_description": THEME_DESCRIPTIONS[theme],
            })
            return result.get("document", "")
        except Exception as e:
            print(f"  ✗ Distill update error ({theme}): {e}")
            return current

    # ── Storage ────────────────────────────────────────────────────

    def _load(self, speaker: str, theme: str) -> str:
        raw = self.store.get_summary(f"distilled:{theme}:{speaker}")
        if not raw:
            return ""
        return self._strip_header(raw)

    def _save(self, speaker: str, theme: str, content: str):
        self.store.save_summary(
            f"distilled:{theme}:{speaker}",
            f"Distilled {theme.title()} — {speaker}",
            content,
            {"speaker": speaker, "theme": theme},
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _get(self, identifier: str) -> str:
        raw = self.store.get_summary(identifier)
        if not raw:
            return ""
        return self._strip_header(raw)

    def _all_docs(self):
        if self.store._summary_store is None:
            return []
        return list(self.store._summary_store.docstore._dict.values())

    @staticmethod
    def _strip_header(text: str) -> str:
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else text


def _esc(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")
