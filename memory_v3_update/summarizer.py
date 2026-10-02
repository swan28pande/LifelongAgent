"""
Hierarchical summarizer: weekly → monthly → yearly → lifetime.

Key design:
- Weekly: raw daily conversations + per-domain preference change sequences
  (transition-only, may have noise) → narrative with observed preferences + facts.
- Monthly: weekly summaries + per-domain sequences → speculative patterns (analyzed independently per domain) + accumulated facts.
- Yearly: monthly summaries → confirmed patterns (analyzed independently per domain) + all facts.
- Lifetime (global blueprint): yearly summaries only → complete user profile with
  confirmed patterns for every domain and a full fact sheet about the user.

Each level feeds the next. Pattern reasoning is isolated by domain to prevent interference.

Incremental updates: `update_after_ingest(date)` rebuilds only the summary chain
affected by a newly ingested day — the containing week, its month, year, and lifetime.
Each level is skipped if its source data hash hasn't changed, so re-ingesting the same
day or ingesting a second day in the same week only rebuilds what actually changed.
"""

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .store import MemoryStore
from .prompts import (
    DOMAIN_DISCOVERY_SYSTEM, LIFETIME_SYSTEM, MONTHLY_FACTS_SYSTEM, MONTHLY_PATTERN_SYSTEM,
    WEEKLY_FACTS_SYSTEM, WEEKLY_PATTERN_SYSTEM, YEARLY_CONFIRMATION_SYSTEM, YEARLY_FACTS_SYSTEM,
)


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


def _esc(s: str) -> str:
    """Escape curly braces so LangChain won't treat them as template variables."""
    return s.replace("{", "{{").replace("}", "}}")


def _fmt_change_seq(pairs: List[Tuple[str, str]]) -> str:
    """
    Emit one entry per value transition, including the duration of the phase.
    e.g. "oat milk latte(Sunday Mar-01-2026) [held for 7 days] → black coffee(Sunday Mar-08-2026)"
    """
    if not pairs:
        return ""
    
    # First, get all transitions
    transitions = []
    prev_val = None
    for date, val in pairs:
        if val != prev_val:
            transitions.append({"date": date, "val": val})
            prev_val = val
            
    if not transitions:
        return ""

    result = []
    for i in range(len(transitions)):
        curr = transitions[i]
        dt = datetime.strptime(curr["date"], "%Y-%m-%d")
        fmt_date = dt.strftime('%A %b-%d-%Y')
        
        if i < len(transitions) - 1:
            next_date = transitions[i+1]["date"]
            d1 = datetime.strptime(curr["date"], "%Y-%m-%d")
            d2 = datetime.strptime(next_date, "%Y-%m-%d")
            days = (d2 - d1).days
            result.append(f"{curr['val']}({fmt_date}) [held for {days} days]")
        else:
            result.append(f"{curr['val']}({fmt_date}) [current]")
            
    return " → ".join(result)


class Summarizer:
    def __init__(self, store: MemoryStore, model: str = "gpt-5.5", llm=None):
        self.store = store
        if llm is not None:
            self.llm = llm
        elif model:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, temperature=0.2)
        else:
            self.llm = None
        self._cache: Dict[str, str] = {}
        self._domain_cache: Dict[str, List[str]] = {}  # speaker → discovered domains

    # ── Public pipeline ─────────────────────────────────────────────

    def run(self, force: bool = False):
        """Full pipeline per speaker: weekly → monthly → yearly → lifetime."""
        start, end = self.store.get_date_range()
        if not start:
            print("No memories found — nothing to summarize.")
            return
        # Normalise partial dates (e.g. '2022' or '2022-05') to full YYYY-MM-DD
        if len(start) == 4:
            start = f"{start}-01-01"
        elif len(start) == 7:
            start = f"{start}-01"
        if len(end) == 4:
            end = f"{end}-12-31"
        elif len(end) == 7:
            end = _month_bounds(end)[1]

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

    def update_after_ingest(self, date: str):
        """
        Incremental update: rebuild only the summary chain affected by `date`.

        After ingesting a day, this rebuilds the containing week, then cascades
        upward through month → year → lifetime — but only when the source data
        for that level has actually changed (checked via a content hash stored
        in summary_meta). If a week's conversations and preferences haven't
        changed since it was last summarized, the week is skipped and so is
        everything above it.
        """
        speakers = self.store.get_all_speakers() or ["user"]
        week_id = _iso_week(date)
        month_id = date[:7]
        year_id = date[:4]

        for speaker in speakers:
            changed = self._build_if_dirty_weekly(week_id, speaker)
            if not changed:
                continue
            changed = self._build_if_dirty_monthly(month_id, speaker)
            if not changed:
                continue
            changed = self._build_if_dirty_yearly(year_id, speaker)
            if not changed:
                continue
            self.lifetime(speaker=speaker, force=True)

    # ── Hash-based dirty checks ────────────────────────────────────

    def _weekly_source_hash(self, week_id: str, speaker: str) -> str:
        start, end = _week_bounds(week_id)
        docs = self.store.get_conversations_by_date_range(start, end)
        mems = self.store.query_memories(speaker=speaker, start_date=start, end_date=end, limit=5000)
        content = "".join(d.page_content for d in docs) + "|" + str([(p["entity"], p["content"], p["date"], p.get("type", "preference")) for p in mems])
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _monthly_source_hash(self, month_id: str, speaker: str) -> str:
        start, end = _month_bounds(month_id)
        weeks = self._all_weeks(start, end)
        parts = [self._cached(f"week:{wk}:{speaker}") or "" for wk in weeks]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _yearly_source_hash(self, year: str, speaker: str) -> str:
        months = self._all_months(f"{year}-01-01", f"{year}-12-31")
        parts = [self._cached(f"month:{mo}:{speaker}") or "" for mo in months]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _build_if_dirty_weekly(self, week_id: str, speaker: str) -> bool:
        identifier = f"week:{week_id}:{speaker}"
        current_hash = self._weekly_source_hash(week_id, speaker)
        stored_hash = self.store.get_summary_hash(identifier)
        if stored_hash == current_hash:
            return False
        self.weekly(week_id, speaker=speaker, force=True)
        self.store.set_summary_hash(identifier, current_hash)
        return True

    def _build_if_dirty_monthly(self, month_id: str, speaker: str) -> bool:
        identifier = f"month:{month_id}:{speaker}"
        current_hash = self._monthly_source_hash(month_id, speaker)
        stored_hash = self.store.get_summary_hash(identifier)
        if stored_hash == current_hash:
            return False
        self.monthly(month_id, speaker=speaker, force=True)
        self.store.set_summary_hash(identifier, current_hash)
        return True

    def _build_if_dirty_yearly(self, year: str, speaker: str) -> bool:
        identifier = f"year:{year}:{speaker}"
        current_hash = self._yearly_source_hash(year, speaker)
        stored_hash = self.store.get_summary_hash(identifier)
        if stored_hash == current_hash:
            return False
        self.yearly(year, speaker=speaker, force=True)
        self.store.set_summary_hash(identifier, current_hash)
        return True

    # ── Summary levels ───────────────────────────────────────────────

    def weekly(self, week_id: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"week:{week_id}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end    = _week_bounds(week_id)
        conversations = self._get_weekly_conversations(start, end)
        if not conversations:
            return ""

        # 1. Independent Domain Analysis
        domains = self._get_recurring_domains(speaker)
        domain_patterns = {}
        
        pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", WEEKLY_PATTERN_SYSTEM),
            ("human", f"Entity: {{entity}}\nTimeline:\n{{transitions}}")
        ])

        for ent in sorted(domains):
            pairs = self.store.get_preference_sequence(ent, start, end, speaker=speaker)
            seq   = _fmt_change_seq(pairs)
            if not seq: continue

            try:
                print(f"    - Weekly analysis for domain: {ent}")
                chain = pattern_prompt | self.llm | JsonOutputParser()
                domain_patterns[ent] = chain.invoke({"speaker": speaker, "entity": ent, "transitions": seq})
            except Exception as e:
                print(f"      Weekly pattern error for {ent}: {e}")

        # 2. Pull structured facts and events from DB, generate narrative
        facts = self.store.query_memories(
            speaker=speaker, start_date=start, end_date=end,
            type="fact", limit=200,
        )
        events = self.store.query_memories(
            speaker=speaker, start_date=start, end_date=end,
            type="event", limit=200,
        )

        facts_block = "\n".join(
            f"[{f['date']}] ({f['entity']}) {f['content']}" for f in facts
        ) if facts else "(no new facts this week)"
        events_block = "\n".join(
            f"[{e['date']}] ({e['entity']}) {e['content']}" for e in events
        ) if events else "(no events this week)"

        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", WEEKLY_FACTS_SYSTEM),
            ("human",
             f"Week: {week_id}\n\n"
             f"FACTS:\n{_esc(facts_block)}\n\n"
             f"EVENTS:\n{_esc(events_block)}")
        ])

        print(f"    - Generating weekly narrative for {week_id}")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"speaker": speaker})
        except Exception as e:
            print(f"      Weekly narrative error: {e}")
            facts_res = {"narrative": "No details found."}

        fact_list = list(dict.fromkeys(f"{f['entity']}: {f['content']}" for f in facts))
        event_list = list(dict.fromkeys(f"[{e['date']}] {e['content']}" for e in events))

        # 3. Merge into readable prose
        narrative = facts_res.get("narrative", "")
        sections = [narrative] if narrative else []
        if fact_list:
            sections.append("Key facts: " + "; ".join(fact_list))
        if event_list:
            sections.append("Events: " + "; ".join(event_list))
        if domain_patterns:
            pat_lines = []
            for ent, pat in domain_patterns.items():
                cycle = pat.get("potential_cycle") or pat.get("identified_pattern", "")
                if cycle and cycle.lower() != "no pattern":
                    pat_lines.append(f"{ent}: {cycle}")
            if pat_lines:
                sections.append("Patterns: " + "; ".join(pat_lines))

        combined_summary = "\n\n".join(sections) if sections else "No details found."
        title = f"Weekly Summary for {week_id}"

        self.store.save_summary(identifier, title, combined_summary, {"speaker": speaker})
        self._cache[identifier] = combined_summary
        print(f"  ✓ Summary saved: {identifier}")
        return combined_summary

    def monthly(self, month_id: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"month:{month_id}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = _month_bounds(month_id)
        weeks      = self._all_weeks(start, end)
        weekly_summaries = [
            (wk, self._cached(f"week:{wk}:{speaker}"))
            for wk in weeks
            if self._cached(f"week:{wk}:{speaker}")
        ]
        if not weekly_summaries:
            return ""

        weekly_block = "\n\n".join(
            f"[{wk}]\n{summary}" for wk, summary in weekly_summaries
        )

        # 1. Independent Pattern Speculation per Domain
        domains = self._get_recurring_domains(speaker)
        pattern_speculation = {}
        
        pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", MONTHLY_PATTERN_SYSTEM),
            ("human",
             f"Entity: {{entity}}\n"
             f"Month: {month_id} ({start} to {end})\n\n"
             f"WEEKLY OBSERVATIONS:\n{_esc(weekly_block)}\n\n"
             f"TRANSITION TIMELINE:\n{{transitions}}")
        ])

        for ent in sorted(domains):
            pairs = self.store.get_preference_sequence(ent, start, end, speaker=speaker)
            seq   = _fmt_change_seq(pairs)
            if not seq:
                continue

            print(f"    - Analyzing domain: {ent}")
            try:
                chain = pattern_prompt | self.llm | JsonOutputParser()
                res   = chain.invoke({"speaker": speaker, "entity": ent, "transitions": seq})
                pattern_speculation[ent] = res
            except Exception as e:
                print(f"      Pattern error for {ent}: {e}")

        # 2. Consolidate Facts and Events from DB
        facts = self.store.query_memories(
            speaker=speaker, start_date=start, end_date=end,
            type="fact", limit=500,
        )
        events = self.store.query_memories(
            speaker=speaker, start_date=start, end_date=end,
            type="event", limit=500,
        )

        facts_block = "\n".join(
            f"[{f['date']}] ({f['entity']}) {f['content']}" for f in facts
        ) if facts else "(no facts this month)"
        events_block = "\n".join(
            f"[{e['date']}] ({e['entity']}) {e['content']}" for e in events
        ) if events else "(no events this month)"

        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", MONTHLY_FACTS_SYSTEM),
            ("human",
             f"Month: {month_id}\n\n"
             f"FACTS:\n{_esc(facts_block)}\n\n"
             f"EVENTS:\n{_esc(events_block)}")
        ])

        print(f"    - Consolidating facts for {month_id}...")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"speaker": speaker})
            print(f"    ✓ Facts consolidated for {month_id}")
        except Exception as e:
            print(f"      Facts error: {e}")
            facts_res = {"facts": [], "events": []}

        # 3. Save as readable prose
        narrative = facts_res.get("narrative", "")
        sections = [narrative] if narrative else []
        consolidated_facts = facts_res.get("facts", [])
        consolidated_events = facts_res.get("events", [])
        if consolidated_facts:
            sections.append("Key facts: " + "; ".join(consolidated_facts))
        if consolidated_events:
            sections.append("Events: " + "; ".join(consolidated_events))
        if pattern_speculation:
            pat_lines = []
            for ent, pat in pattern_speculation.items():
                rule = pat.get("identified_pattern", "")
                if rule and rule.lower() != "no pattern" and rule.lower() != "none":
                    pat_lines.append(f"{ent}: {rule}")
            if pat_lines:
                sections.append("Patterns: " + "; ".join(pat_lines))

        combined_summary = "\n\n".join(sections) if sections else "No details found."
        title = f"Monthly Summary for {month_id}"

        print(f"    - Saving monthly summary for {month_id}")
        self.store.save_summary(identifier, title, combined_summary, {"speaker": speaker})
        self._cache[identifier] = combined_summary
        print(f"  ✓ Monthly summary saved: {identifier}")
        return combined_summary

    def yearly(self, year: str, speaker: str = "user", force: bool = False) -> str:
        identifier = f"year:{year}:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        months = self._all_months(f"{year}-01-01", f"{year}-12-31")
        monthly_summaries = [
            (mo, self._cached(f"month:{mo}:{speaker}"))
            for mo in months
            if self._cached(f"month:{mo}:{speaker}")
        ]
        if not monthly_summaries:
            return ""

        monthly_block = "\n\n".join(
            f"[{mo}]\n{summary}" for mo, summary in monthly_summaries
        )

        # 1. Independent Confirmation per Domain
        domains = self._get_recurring_domains(speaker)
        confirmed_patterns = {}
        
        confirmation_prompt = ChatPromptTemplate.from_messages([
            ("system", YEARLY_CONFIRMATION_SYSTEM),
            ("human",
             f"Entity: {{entity}}\n"
             f"Year: {year}\n\n"
             f"MONTHLY SPECULATIONS:\n{_esc(monthly_block)}")
        ])

        for ent in sorted(domains):
            print(f"    - Confirming domain: {ent}")
            try:
                chain = confirmation_prompt | self.llm | JsonOutputParser()
                res   = chain.invoke({"speaker": speaker, "entity": ent})
                confirmed_patterns[ent] = res
            except Exception as e:
                print(f"      Confirmation error for {ent}: {e}")

        # 2. Accumulate Facts and Events (Final Consolidation)
        facts = self.store.query_memories(
            speaker=speaker, start_date=f"{year}-01-01", end_date=f"{year}-12-31",
            type="fact", limit=2000,
        )
        events = self.store.query_memories(
            speaker=speaker, start_date=f"{year}-01-01", end_date=f"{year}-12-31",
            type="event", limit=2000,
        )

        facts_block = "\n".join(
            f"[{f['date']}] ({f['entity']}) {f['content']}" for f in facts
        ) if facts else "(no facts this year)"
        events_block = "\n".join(
            f"[{e['date']}] ({e['entity']}) {e['content']}" for e in events
        ) if events else "(no events this year)"

        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", YEARLY_FACTS_SYSTEM),
            ("human",
             f"Year: {year}\n\n"
             f"FACTS:\n{_esc(facts_block)}\n\n"
             f"EVENTS:\n{_esc(events_block)}")
        ])

        print(f"    - Finalizing yearly profile for {year}")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"speaker": speaker})
        except Exception as e:
            print(f"      Facts error: {e}")
            facts_res = {}

        # 3. Save as readable prose
        sections = []
        yearly_facts = facts_res.get("facts", [])
        yearly_events = facts_res.get("events", [])
        if yearly_facts:
            sections.append("Facts: " + "; ".join(yearly_facts))
        if yearly_events:
            sections.append("Events: " + "; ".join(yearly_events))
        if confirmed_patterns:
            pat_lines = []
            for ent, pat in confirmed_patterns.items():
                rule = pat.get("confirmed_rule", "")
                conf = pat.get("confidence", "")
                if rule and rule.lower() not in ("no pattern", "none"):
                    pat_lines.append(f"{ent}: {rule} ({conf})")
            if pat_lines:
                sections.append("Patterns: " + "; ".join(pat_lines))

        combined_summary = "\n\n".join(sections) if sections else "No details found."
        title = f"Yearly Summary for {year}"

        self.store.save_summary(identifier, title, combined_summary, {"speaker": speaker})
        self._cache[identifier] = combined_summary
        print(f"  ✓ Summary saved: {identifier}")
        return combined_summary

    def lifetime(self, speaker: str = "user", force: bool = False) -> str:
        identifier = f"lifetime:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = self.store.get_date_range()
        if not start:
            return ""

        # The stored bounds may be partial dates; only their years are needed here.
        years = [str(year) for year in range(int(start[:4]), int(end[:4]) + 1)]
        yearly_summaries = [
            (yr, self._cached(f"year:{yr}:{speaker}"))
            for yr in years
            if self._cached(f"year:{yr}:{speaker}")
        ]
        if not yearly_summaries:
            return ""

        yearly_block = "\n\n".join(
            f"[{yr}]\n{summary}" for yr, summary in yearly_summaries
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", LIFETIME_SYSTEM),
            ("human",
             f"Speaker: {speaker} | Full date range: {start} to {end}\n\n"
             f"YEARLY SUMMARIES:\n{_esc(yearly_block)}")
        ])
        return self._run_and_save(prompt, identifier, f"lifetime:{speaker}",
                                  extra_meta={"speaker": speaker}, variables={"speaker": speaker})

    # ── Memory formatters ────────────────────────────────────────────

    def _get_weekly_conversations(self, start: str, end: str) -> str:
        """Raw conversation text for the week, grouped and labelled by date."""
        docs = self.store.get_conversations_by_date_range(start, end)
        if not docs:
            return ""
        by_date: Dict[str, List[str]] = defaultdict(list)
        for doc in docs:
            by_date[doc.metadata.get("source_date", "")].append(doc.page_content)
        lines = []
        for date in sorted(by_date):
            dt = datetime.strptime(date, "%Y-%m-%d")
            lines.append(f"\n--- {dt.strftime('%A, %B %d %Y')} ---")
            lines.extend(by_date[date])
        return "\n".join(lines)

    def _preference_change_sequences(
        self, start_date: str, end_date: str, speaker: Optional[str] = None
    ) -> str:
        """For each recurring domain, emit a transition-only change sequence."""
        domains = self._get_recurring_domains(speaker)
        lines = []
        for ent in sorted(domains):
            pairs = self.store.get_preference_sequence(ent, start_date, end_date,
                                                       speaker=speaker)
            seq = _fmt_change_seq(pairs)
            if seq:
                lines.append(f"[{ent}] {seq}")
        return "\n".join(lines)

    def _get_recurring_domains(self, speaker: Optional[str] = None) -> List[str]:
        """
        Discover which preference entities are genuine recurring domains by asking
        the LLM to identify patterns from all stored preference memories.
        Result is cached per speaker for the lifetime of this Summarizer instance.
        """
        cache_key = speaker or "all"
        if cache_key in self._domain_cache:
            return self._domain_cache[cache_key]

        all_prefs = self.store.query_memories(speaker=speaker, type="preference", limit=2000)
        if not all_prefs:
            return []

        from collections import Counter
        counts = Counter(m["entity"] for m in all_prefs if m["entity"])

        candidates = {ent: cnt for ent, cnt in counts.items() if cnt >= 2}
        if not candidates:
            return []
        if not candidates:
            self._domain_cache[cache_key] = list(counts.keys())
            return self._domain_cache[cache_key]

        entity_summary = []
        for ent, cnt in sorted(candidates.items(), key=lambda x: -x[1]):
            values = list({m["content"] for m in all_prefs if m["entity"] == ent})[:5]
            entity_summary.append(f"{ent} ({cnt} obs): {', '.join(values)}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", DOMAIN_DISCOVERY_SYSTEM),
            ("human", "Preference entities:\n" + "\n".join(entity_summary)),
        ])
        try:
            chain  = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({})
            domains = result.get("domains", list(candidates.keys()))
            print(f"  Discovered recurring domains for {cache_key}: {domains}")
        except Exception as e:
            print(f"  Domain discovery error: {e}. Using all entities.")
            domains = list(candidates.keys())

        self._domain_cache[cache_key] = [d.lower().strip() for d in domains]
        return self._domain_cache[cache_key]

    # ── Helpers ──────────────────────────────────────────────────────

    def _run_and_save(
        self, prompt, identifier: str, label: str, extra_meta: Dict = {},
        variables: Optional[Dict] = None,
    ) -> str:
        try:
            chain   = prompt | self.llm | JsonOutputParser()
            result  = chain.invoke(variables or {})
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
            days_ahead = (7 - cur.weekday()) % 7 or 7
            cur += timedelta(days=days_ahead)
        weeks.add(_iso_week(end))
        return sorted(weeks)

    @staticmethod
    def _all_months(start: str, end: str) -> List[str]:
        months, cur = [], datetime.strptime(start[:7], "%Y-%m")
        end_dt = datetime.strptime(end[:7], "%Y-%m")
        while cur <= end_dt:
            months.append(cur.strftime("%Y-%m"))
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return sorted(set(months))
