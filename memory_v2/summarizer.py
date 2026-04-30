"""
Hierarchical summarizer: weekly → monthly → yearly → lifetime.

Key design:
- Preferences get temporal sequences injected (date-ordered arrow chains).
- All other types (facts, goals, events, ...) are grouped by type and listed.
- Each level feeds the next, so lifetime is a true compression of everything.
- All summaries are saved to the FAISS summary store for retrieval.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .store import MemoryStore


def _iso_week(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _week_bounds(week_id: str) -> Tuple[str, str]:
    year, w = week_id.split("-W")
    monday = datetime.strptime(f"{year} {int(w)} 1", "%G %V %u")
    return monday.strftime("%Y-%m-%d"), (monday + timedelta(days=6)).strftime("%Y-%m-%d")


def _month_bounds(month_id: str) -> Tuple[str, str]:
    y, m = int(month_id[:4]), int(month_id[5:7])
    start = f"{y}-{m:02d}-01"
    end_dt = datetime(y + (m // 12), (m % 12) + 1, 1) - timedelta(days=1)
    return start, end_dt.strftime("%Y-%m-%d")


def _fmt_seq(pairs: List[Tuple[str, str]]) -> str:
    """
    Format [(date, content), ...] as explicit run-length text.

    Output: "First 3 days: espresso (Mar 01–Mar 03) → then 2 days: latte (Mar 04–Mar 05)"
    """
    if not pairs:
        return ""

    # Build runs: (value, [dates])
    runs: List[Tuple[str, List[str]]] = []
    for date, content in pairs:
        if runs and runs[-1][0] == content:
            runs[-1][1].append(date)
        else:
            runs.append((content, [date]))

    parts = []
    for i, (value, dates) in enumerate(runs):
        n = len(dates)
        start = datetime.strptime(dates[0],  "%Y-%m-%d").strftime("%b %d")
        end   = datetime.strptime(dates[-1], "%Y-%m-%d").strftime("%b %d")
        date_range = start if n == 1 else f"{start}–{end}"
        day_word   = "day" if n == 1 else "days"
        prefix     = "First" if i == 0 else "then"
        parts.append(f"{prefix} {n} {day_word}: {value.rstrip('.')} ({date_range})")

    return " → ".join(parts)


class Summarizer:
    def __init__(self, store: MemoryStore, model: str = "gpt-4o-mini"):
        self.store = store
        self.llm   = ChatOpenAI(model=model, temperature=0.2)
        # Persisted summary text (loaded lazily from FAISS store)
        self._cache: Dict[str, str] = {}

    # ── Public pipeline ─────────────────────────────────────────────

    def run(self, force: bool = False):
        """Full pipeline per speaker: weekly → monthly → yearly → lifetime."""
        start, end = self.store.get_date_range()
        if not start:
            print("No memories found — nothing to summarize.")
            return

        speakers = self.store.get_all_speakers() or ["user"]
        weeks    = self._all_weeks(start, end)
        months   = self._all_months(start, end)
        years    = sorted({m[:4] for m in months})

        for speaker in speakers:
            print(f"  Summarising speaker: {speaker}")
            for wk in weeks:
                self.weekly(wk, speaker=speaker, force=force)
            for mo in months:
                self.monthly(mo, speaker=speaker, force=force)
            for yr in years:
                self.yearly(yr, speaker=speaker, force=force)
            self.lifetime(speaker=speaker, force=force)

    def weekly(self, week_id: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"week:{week_id}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = _week_bounds(week_id)
        memories   = self.store.query_memories(speaker=speaker, start_date=start, end_date=end, limit=500)
        if not memories:
            return ""

        temporal_block = self._preference_sequences(memories, start, end, speaker)
        other_block    = self._non_preference_block(memories)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a memory analyst writing a weekly summary for {speaker}.\n\n"
             "RULES FOR PREFERENCES:\n"
             "- The preference sequences below are exact ground truth — do not paraphrase or omit runs.\n"
             "- For each entity reproduce the exact run-length pattern: "
             "'First N days: X → then M days: Y → ...'.\n\n"
             "RULES FOR OTHER MEMORIES:\n"
             "- Group facts, goals, events, routines naturally into the narrative.\n\n"
             "Write a single clear paragraph that leads with the preference runs, "
             "then weaves in the other memories."),
            ("human",
             f"Speaker: {speaker} | Week: {week_id}\n\n"
             f"PREFERENCE SEQUENCES:\n{temporal_block or 'None'}\n\n"
             f"OTHER MEMORIES:\n{other_block or 'None'}\n\n"
             "Return JSON: {{\"title\": \"...\", \"summary\": \"single paragraph\"}}"),
        ])
        return self._run_and_save(prompt, identifier, week_id,
                                  extra_meta={"month_id": week_id[:7], "speaker": speaker})

    def monthly(self, month_id: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"month:{month_id}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = _month_bounds(month_id)
        memories   = self.store.query_memories(speaker=speaker, start_date=start, end_date=end, limit=2000)
        if not memories:
            return ""

        other_block = self._non_preference_block(memories)
        weeks = self._all_weeks(start, end)
        weekly_pref_block = self._per_week_preference_sequences(weeks, speaker)
        weekly_narrative  = "\n\n".join(
            f"{wk}: {self._cached(f'week:{wk}:{speaker}')}"
            for wk in weeks
            if self._cached(f"week:{wk}:{speaker}")
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a memory analyst writing a monthly summary for {speaker}.\n\n"
             "RULES FOR PREFERENCES:\n"
             "- Per-week preference sequences are exact ground truth.\n"
             "- Reproduce the full month week-by-week: "
             "'Week N: First X days: espresso → then Y days: latte | Week N+1: ...'\n\n"
             "RULES FOR OTHER MEMORIES:\n"
             "- Weave facts, goals, and events naturally.\n\n"
             "Write a single clear paragraph leading with per-week preference runs."),
            ("human",
             f"Speaker: {speaker} | Month: {month_id}\n\n"
             f"PER-WEEK PREFERENCE SEQUENCES:\n{weekly_pref_block or 'None'}\n\n"
             f"OTHER MEMORIES:\n{other_block or 'None'}\n\n"
             f"WEEKLY SUMMARIES (context):\n{weekly_narrative or 'None'}\n\n"
             "Return JSON: {{\"title\": \"...\", \"summary\": \"single paragraph\"}}"),
        ])
        return self._run_and_save(prompt, identifier, month_id,
                                  extra_meta={"speaker": speaker})

    def yearly(self, year: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"year:{year}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = f"{year}-01-01", f"{year}-12-31"
        memories   = self.store.query_memories(speaker=speaker, start_date=start, end_date=end, limit=5000)
        if not memories:
            return ""

        temporal_block = self._preference_sequences(memories, start, end, speaker)
        other_block    = self._non_preference_block(memories)

        unique_months = sorted({m["date"][:7] for m in memories if m["date"]})
        monthly_ctx = "\n\n".join(
            f"{mo}: {self._cached(f'month:{mo}:{speaker}')}"
            for mo in unique_months
            if self._cached(f"month:{mo}:{speaker}")
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a memory analyst. Summarise {speaker}'s year. "
             "Identify multi-month preference arcs and major life events."),
            ("human",
             f"Speaker: {speaker} | Year: {year}\n\n"
             f"PREFERENCE SEQUENCES:\n{temporal_block or 'None'}\n\n"
             f"OTHER MEMORIES:\n{other_block or 'None'}\n\n"
             f"MONTHLY SUMMARIES:\n{monthly_ctx or 'None'}\n\n"
             "Return JSON: {{\"title\": \"...\", \"summary\": \"single paragraph\"}}"),
        ])
        return self._run_and_save(prompt, identifier, year,
                                  extra_meta={"speaker": speaker})

    def lifetime(self, speaker: str = "user", force: bool = False) -> str:
        identifier = f"lifetime:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = self.store.get_date_range()
        if not start:
            return ""

        memories       = self.store.query_memories(speaker=speaker, limit=10000)
        temporal_block = self._preference_sequences(memories, start, end, speaker)
        other_block    = self._non_preference_block(memories)

        years = sorted({m["date"][:4] for m in memories if m["date"]})
        yearly_ctx = "\n\n".join(
            f"{yr}: {self._cached(f'year:{yr}:{speaker}')}"
            for yr in years
            if self._cached(f"year:{yr}:{speaker}")
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are the ultimate memory archivist. Create a lifetime summary for {speaker} "
             "capturing who they are: stable facts, evolving preferences, long-term goals, "
             "and the grand narrative of their life so far. "
             "This will be injected at the top of every future conversation."),
            ("human",
             f"Speaker: {speaker} | Date range: {start} to {end}\n\n"
             f"PREFERENCE SEQUENCES:\n{temporal_block or 'None'}\n\n"
             f"OTHER MEMORIES:\n{other_block or 'None'}\n\n"
             f"YEARLY SUMMARIES:\n{yearly_ctx or 'None'}\n\n"
             "Return JSON: {{\"title\": \"...\", \"summary\": \"single paragraph\"}}"),
        ])
        return self._run_and_save(prompt, identifier, f"lifetime:{speaker}",
                                  extra_meta={"speaker": speaker})

    # ── Memory formatters ───────────────────────────────────────────

    def _preference_sequences(
        self,
        memories: List[Dict],
        start_date: str,
        end_date: str,
        speaker: Optional[str] = None,
    ) -> str:
        pref_subjects = sorted({
            m["subject"] for m in memories
            if m["type"] == "preference" and m["subject"]
        })
        lines = []
        for subj in pref_subjects:
            pairs = self.store.get_preference_sequence(subj, start_date, end_date,
                                                       speaker=speaker)
            if pairs:
                lines.append(f"[{subj}] {_fmt_seq(pairs)}")
        return "\n".join(lines)

    def _per_week_preference_sequences(self, weeks: List[str],
                                       speaker: Optional[str] = None) -> str:
        """Per-week preference sequences for monthly summaries."""
        lines = []
        for wk in weeks:
            start, end = _week_bounds(wk)
            pref_subjects = sorted(set(
                m["subject"]
                for m in self.store.query_memories(
                    type="preference", speaker=speaker,
                    start_date=start, end_date=end)
                if m["subject"]
            ))
            if not pref_subjects:
                continue
            lines.append(f"Week {wk}:")
            for subj in pref_subjects:
                pairs = self.store.get_preference_sequence(subj, start, end,
                                                           speaker=speaker)
                if pairs:
                    lines.append(f"  [{subj}] {_fmt_seq(pairs)}")
        return "\n".join(lines)

    def _non_preference_block(self, memories: List[Dict]) -> str:
        """Group all non-preference memories by type, list contents."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for m in memories:
            if m["type"] != "preference":
                groups[m["type"] or "general"].append(m["content"])

        lines = []
        for mtype in sorted(groups):
            lines.append(f"[{mtype}]")
            for content in groups[mtype]:
                lines.append(f"  - {content}")
        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────

    def _run_and_save(
        self, prompt, identifier: str, label: str, extra_meta: Dict = {}
    ) -> str:
        try:
            chain  = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({})
            title   = result.get("title", label)
            content = result.get("summary", "")

            self.store.save_summary(identifier, title, content, extra_meta)
            self._cache[identifier] = content
            print(f"  ✓ Summary saved: {identifier}")
            return content
        except Exception as e:
            print(f"  ✗ Summary error for {identifier}: {e}")
            return ""

    def _cached(self, identifier: str) -> Optional[str]:
        if identifier not in self._cache:
            text = self.store.get_summary(identifier)
            if text:
                # Strip the "[identifier] Title\n\n" prefix stored by save_summary
                parts = text.split("\n\n", 1)
                self._cache[identifier] = parts[1] if len(parts) > 1 else text
        return self._cache.get(identifier)

    @staticmethod
    def _all_weeks(start: str, end: str) -> List[str]:
        """Return all ISO week IDs that overlap [start, end]."""
        weeks: set = set()
        cur = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        while cur <= end_dt:
            weeks.add(_iso_week(cur.strftime("%Y-%m-%d")))
            # Jump to next Monday (or stay if already Monday)
            days_ahead = (7 - cur.weekday()) % 7 or 7
            cur += timedelta(days=days_ahead)
        weeks.add(_iso_week(end))  # always include end date's week
        return sorted(weeks)

    @staticmethod
    def _all_months(start: str, end: str) -> List[str]:
        months, cur = [], datetime.strptime(start[:7], "%Y-%m")
        end_dt = datetime.strptime(end[:7], "%Y-%m")
        while cur <= end_dt:
            months.append(cur.strftime("%Y-%m"))
            # advance by one month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return sorted(set(months))
