"""
Knowledge distillation from hierarchical summaries.

The summarizer builds temporal snapshots (weekly → monthly → yearly → lifetime).
The distiller reasons over those snapshots to produce focused knowledge
documents that track temporal evolution, one per extraction type:

  preferences — recurring choices and how they shift over time
  facts       — stable attributes and their transitions with reasoning
  events      — chronological milestones and key occurrences

Each document records transitions with dates and reasoning:

    "Priya and Rohan: close friends who run together (since Mar 2026).
     Relationship cooled after the project disagreement (Jul 2026)."

These sit at the top of the memory pyramid — derived from extensive
reasoning over summaries, not simple aggregation.

    raw memories (DB)
        → hierarchical summaries (weekly/monthly/yearly/lifetime)
            → distilled knowledge documents (reasoned, multi-theme)

The documents are updated continually: after each day's ingest rebuilds the
affected summaries, only the changed week/month summaries are fed to the
distiller. It reads the existing document and merges in new information —
never rebuilds from scratch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .store import MemoryStore
from .prompts import DISTILL_SYSTEM, DISTILL_UPDATE_SYSTEM, THEME_DESCRIPTIONS


THEMES = ("preferences", "facts", "events")


def _iso_week(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


class KnowledgeDistiller:
    """Distills hierarchical summaries into themed knowledge documents."""

    def __init__(self, store: MemoryStore, llm):
        self.store = store
        self.llm = llm

    # ── Incremental update (after each day's ingest) ──────────────

    def update_after_ingest(self, speaker: str, date: str) -> Dict[str, str]:
        """
        Incrementally update knowledge documents after a day's summaries
        have been rebuilt.

        Only reads the affected week and month summaries — not the entire
        hierarchy — and merges new information into existing documents.
        If no documents exist yet, falls back to full creation.
        """
        has_existing = any(self._load(speaker, t) for t in THEMES)
        if not has_existing:
            return self.distill(speaker)

        new_summaries = self._gather_changed(speaker, date)
        if not new_summaries:
            return {}

        results = {}
        for theme in THEMES:
            current = self._load(speaker, theme)
            if current:
                updated = self._update(speaker, theme, current, new_summaries)
            else:
                all_summaries = self._gather_all(speaker)
                updated = self._create(speaker, theme, all_summaries)

            if updated:
                self._save(speaker, theme, updated)
                results[theme] = updated
                print(f"  ✓ Updated: {theme}:{speaker}")

        return results

    def _gather_changed(self, speaker: str, date: str) -> str:
        """Collect only the summaries affected by a specific date."""
        sections = []
        week_id = _iso_week(date)
        month_id = date[:7]

        week = self._get(f"week:{week_id}:{speaker}")
        if week:
            sections.append(f"=== WEEK {week_id} ===\n{week}")

        month = self._get(f"month:{month_id}:{speaker}")
        if month:
            sections.append(f"=== MONTH {month_id} ===\n{month}")

        return "\n\n".join(sections)

    # ── Full distillation (initial build or explicit rebuild) ─────

    def distill(self, speaker: str) -> Dict[str, str]:
        """
        Create or update all knowledge documents for a speaker using
        the full summary hierarchy.
        """
        summaries = self._gather_all(speaker)
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

    def _gather_all(self, speaker: str) -> str:
        """Collect all available summaries for a speaker."""
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
