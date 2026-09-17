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
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


DOMAIN_DISCOVERY_SYSTEM = """\
You are a memory analyst. Given a list of all preference memories (entity + value pairs),
identify which entities represent genuine RECURRING entities —
things the person chooses regularly from a consistent category over time.

A recurring entity has multiple observations across different dates with a repeating
pattern of values.

Exclude entities that are:
- Too granular (a specific detail of a larger entity)
- One-off or rarely mentioned (fewer than 3 observations)
- Not a stable choice category

Return ONLY a JSON object: {{"domains": ["entity1", "entity2", ...]}}
List only the canonical recurring entities.
"""

WEEKLY_PATTERN_SYSTEM = """\
You are a mathematical pattern analyst. Analyze the preference transition sequence for a SINGLE entity for {speaker} during one week.

1. LOG: List exactly what changed and when. 
   Format: [Date] Value (Duration: X days)
2. SPECULATE: Does this sequence suggest a potential N-day cycle or a calendar-based rhythm (e.g. Mon-Fri)? 
3. PRECISION: Avoid vague terms like "usually" or "tends to". Use specific durations.

Return JSON:
{{
  "mathematical_sequence": "The exact date-to-date sequence of values and durations",
  "potential_cycle": "Speculate on a possible periodicity (e.g. 'Alternates every 3 days')",
  "reasoning": "Show your calculation of the days between switches."
}}
"""

WEEKLY_FACTS_SYSTEM = """\
You are a facts extractor. Given raw daily conversations for a week for {speaker}, extract all factual information.

Return JSON:
{{
  "facts": ["list of factual statements about the person"],
  "narrative": "A short paragraph summarizing the week's events."
}}
"""

MONTHLY_PATTERN_SYSTEM = """\
You are a mathematical pattern analyst. Analyze the observations for a SINGLE preference entity for {speaker} across several weeks.

Goal: Identify the exact repeating structure for this entity — it may be a fixed N-day alternation, a day-of-week rule, or another repeating structure.

1. CONSOLIDATE: Merge the weekly transition logs into one continuous timeline.
2. CALCULATE: Find the exact number of days the user held each preference value.
3. GROUPING: Group similar values or sub-categories into their primary underlying states (e.g., grouping 'blue pens', 'black pens', and 'red pens' under a single 'pen' state, and comparing that to 'pencil') before calculating cycle lengths.
4. DETECT PATTERN: Look for ANY repeating structure:
   - Fixed N-day alternation (e.g. 3 days of State A then 3 days of State B, totaling a 6-day cycle)
   - Day-of-week rule (e.g. always Tue/Thu for State A)
   - Or any other observable regularity.
5. PRECISION: State the exact START DATE. List values explicitly. Do NOT use vague terms like "usually" or "tends to".

Return JSON:
{{
  "consolidated_timeline": "The full date-to-date sequence for the month, listing each value with its exact date range",
  "identified_pattern": "The definitive repeating rule with start date (e.g. 'Alternates every 7 days: [Value A] from YYYY-MM-DD to YYYY-MM-DD, then [Value B]...')",
  "frequency": "Exact description (e.g. 'Every 3 days', 'Mon/Wed/Fri', 'Every 7 days')",
  "reasoning": "Show the calculation behind your pattern detection."
}}
"""

MONTHLY_FACTS_SYSTEM = """\
You are a facts extractor. Given weekly summaries for {speaker}, consolidate all factual information for the month.

Return JSON:
{{
  "facts": ["list of consolidated factual statements"]
}}
"""

YEARLY_CONFIRMATION_SYSTEM = """\
You are a senior senior pattern analyst. Confirm the definitive mathematical rhythm for a SINGLE entity for {speaker} using monthly summaries.

1. COMPARE: Check if the N-day cycle or calendar rhythm is consistent across all months. Group sub-categories/synonyms into their primary underlying states (e.g. grouping 'blue pens' and 'red pens' under a single 'pen' state vs 'pencil') before checking consistency.
2. VALIDATE: If a month reported a "smoothed" general pattern, look back at the consolidated timelines to re-verify the exact periodicity.
3. RULE: Define the definitive rule. 
   Example: "Fixed 7-day alternation: [State A] for 7 days, then [State B] for 7 days, repeating."
   Example: "Fixed 3-day alternation: [State A] for 3 days, then [State B] for 3 days, repeating."

Return JSON:
{{
  "confirmed_rule": "The exact mathematical rhythm (including start date)",
  "frequency": "The verified periodicity",
  "confidence": "CONFIRMED / LIKELY / EMERGING",
  "reasoning": "Show the calculation that proves this pattern is stable across the year."
}}
"""



class Summarizer:
    def __init__(self, store: MemoryStore, model: str = "gpt-5.5", llm=None):
        self.store = store
        if llm is not None:
            self.llm = llm
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, temperature=0.2)
        self._cache: Dict[str, str] = {}
        self._domain_cache: Dict[str, List[str]] = {}  # speaker → discovered domains

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

        # 2. Extract Facts and Narrative
        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", WEEKLY_FACTS_SYSTEM),
            ("human", f"Week: {week_id}\n\nCONVERSATIONS:\n{{conversations}}")
        ])
        
        print(f"    - Extracting weekly facts for {week_id}")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"conversations": _esc(conversations), "speaker": speaker})
        except Exception as e:
            print(f"      Weekly facts error: {e}")
            facts_res = {"facts": {}, "narrative": "No details found."}

        # 3. Merge
        combined = {
            "title": f"Weekly Summary for {week_id}",
            "summary": str({
                "patterns": domain_patterns,
                "facts": facts_res.get("facts", {}),
                "narrative": facts_res.get("narrative", "")
            })
        }
        
        self.store.save_summary(identifier, combined["title"], combined["summary"], {"speaker": speaker})
        self._cache[identifier] = combined["summary"]
        print(f"  ✓ Summary saved: {identifier}")
        return combined["summary"]

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

        # 2. Extract Facts
        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", MONTHLY_FACTS_SYSTEM),
            ("human", f"WEEKLY SUMMARIES:\n{_esc(weekly_block)}")
        ])
        
        print(f"    - Extracting facts for {month_id}...")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"speaker": speaker})
            print(f"    ✓ Facts extracted for {month_id}")
        except Exception as e:
            print(f"      Facts error: {e}")
            facts_res = {"facts": []}

        # 3. Save combined result
        combined = {
            "title": f"Monthly Summary for {month_id}",
            "summary": str({
                "patterns": pattern_speculation,
                "facts": facts_res.get("facts", [])
            })
        }
        
        print(f"    - Saving monthly summary for {month_id}")
        self.store.save_summary(identifier, combined["title"], combined["summary"], {"speaker": speaker})
        self._cache[identifier] = combined["summary"]
        print(f"  ✓ Monthly summary saved: {identifier}")
        return combined["summary"]

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

        # 2. Accumulate Facts (Final Consolidation)
        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Consolidate all factual info for {speaker} for the year into a clean, merged list of facts.\n\n"
             "Return ONLY a JSON object:\n"
             "{{\n"
             "  \"facts\": [\n"
             "    \"fact 1\",\n"
             "    \"fact 2\",\n"
             "    ...\n"
             "  ]\n"
             "}}"),
            ("human", f"MONTHLY SUMMARIES:\n{_esc(monthly_block)}")
        ])
        
        print(f"    - Finalizing yearly profile for {year}")
        try:
            chain = facts_prompt | self.llm | JsonOutputParser()
            facts_res = chain.invoke({"speaker": speaker})
        except Exception as e:
            print(f"      Facts error: {e}")
            facts_res = {}

        # 3. Save combined result
        combined = {
            "title": f"Yearly Summary for {year}",
            "summary": str({
                "patterns": confirmed_patterns,
                "facts": facts_res.get("facts", [])
            })
        }
        
        self.store.save_summary(identifier, combined["title"], combined["summary"], {"speaker": speaker})
        self._cache[identifier] = combined["summary"]
        print(f"  ✓ Summary saved: {identifier}")
        return combined["summary"]

    def lifetime(self, speaker: str = "user", force: bool = False) -> str:
        identifier = f"lifetime:{speaker}"
        if not force and self._cached(identifier):
            return self._cached(identifier)

        start, end = self.store.get_date_range()
        if not start:
            return ""

        years = sorted({m[:4] for m in self._all_months(start, end)})
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
            ("system",
             f"You are building the definitive lifetime blueprint for {speaker}. "
             "This document is injected at the start of every future conversation "
             "as the ground truth about this person.\n\n"
             "Using all yearly summaries, produce a complete, structured profile:\n\n"
             "1. USER FACTS — Everything stable and confirmed about this person:\n"
             "   identity, age, location, job, relationships, lifestyle, diet, "
             "hobbies, personality traits, and any major life events.\n\n"
             "2. PREFERENCE PATTERNS — For each domain, describe the exact repeating structure observed.\n"
             "   It may be a fixed N-day alternation, a day-of-week pattern, or another form of repetition.\n"
             "   Include the start date and the exact rule. State values explicitly. Avoid vague phrases like 'usually' or 'tends to'.\n\n"
             "3. LIFE NARRATIVE — Chronological milestones.\n\n"
             "Return ONLY a JSON object: {{\"title\": \"Full Lifetime Profile\", \"summary\": \"the full structured text profile\"}}"),
            ("human",
             f"Speaker: {speaker} | Full date range: {start} to {end}\n\n"
             f"YEARLY SUMMARIES:\n{_esc(yearly_block)}")
        ])
        return self._run_and_save(prompt, identifier, f"lifetime:{speaker}",
                                  extra_meta={"speaker": speaker})

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

        all_prefs = self.store.query_memories(speaker=speaker, limit=2000)
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
        self, prompt, identifier: str, label: str, extra_meta: Dict = {}
    ) -> str:
        try:
            chain   = prompt | self.llm | JsonOutputParser()
            result  = chain.invoke({})
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
