"""
Ingestion pipeline.

    day's turns
        │
        ▼
    [1] extract        one LLM call  → dated (entity, content) records
        │
        ▼
    [2] consolidate    deterministic → normalize near-duplicate values to canonical spellings
        │
        ▼
    [3] dedupe         deterministic → drop exact repeats of (entity, content, date)
        │
        ▼
    [4] write          deterministic → SQL inserts
        │
        ▼
    [5] index          deterministic → fixed-size chunks into the RAG store

One LLM call per day.

Deduplication is an exact match on (entity, content, date), so it is a set operation
and a SQL lookup rather than a judgement call. It guards two cases: the same choice
extracted twice from one transcript, and a day ingested more than once (a re-run, or
a second flush that restates a choice already recorded earlier that day).

A repeat on a *different* date is never a duplicate. Recurrence is the evidence that
makes a pattern visible, so every occurrence is kept.

Summarization is not part of this. The caller drives it on a cadence, because
rebuilding a summary for a period still in progress is wasted work and can leave prose
that contradicts the raw timeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from memory_v2.store import MemoryStore

from .prompts import EXTRACT_SYSTEM

# Turns per RAG chunk. Matches memory_v2 so retrieval quality stays comparable
# between the two systems in the benchmark.
CHUNK_SIZE = 5

# Ceiling on rows fetched for one entity on one date during the duplicate check.
# A single day should hold a handful at most; this only bounds a pathological store.
MAX_SAME_DAY_ROWS = 50

# How many distinct values to show the extractor per entity, and how many rows to
# scan to collect them. After 60 days the whole table is ~120 tokens, so the cap is
# insurance against a store that has grown far past that rather than a live limit.
MAX_VALUES_SHOWN = 12
VALUE_SCAN_LIMIT = 2000


CONSOLIDATION_THRESHOLD = 0.75


def _char_ratio(a: str, b: str) -> float:
    """Character-level similarity: 2 * common_len / total_len (SequenceMatcher-style)."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _token_jaccard(a: str, b: str) -> float:
    """
    Combined similarity: token-level Jaccard with subset bonus and
    character-level fallback for typos that split tokens.
    """
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    if ta <= tb or tb <= ta:
        jaccard = max(jaccard, 0.85)
    char_sim = _char_ratio(a, b)
    return max(jaccard, char_sim)


@dataclass
class IngestReport:
    """What the pipeline did with one day. Returned so runs can be audited."""

    date: str
    extracted: int = 0
    added: int = 0
    duplicates: int = 0
    consolidated: int = 0
    chunks_indexed: int = 0
    entities: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    type_counts: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.date}] extracted {self.extracted}", f"added {self.added}"]
        if self.type_counts:
            type_str = " ".join(f"{t}={c}" for t, c in sorted(self.type_counts.items()))
            parts.append(f"({type_str})")
        if self.consolidated:
            parts.append(f"consolidated {self.consolidated}")
        if self.duplicates:
            parts.append(f"skipped {self.duplicates} duplicate")
        parts.append(f"{self.chunks_indexed} chunks")
        if self.errors:
            parts.append(f"ERRORS: {'; '.join(self.errors)}")
        return ", ".join(parts)


class IngestionPipeline:
    def __init__(self, store: MemoryStore, llm):
        self.store = store
        self.llm = llm

    # ── Entry point ─────────────────────────────────────────────────

    def run(
        self, date: str, conversations: List[Dict], speaker: str = "user"
    ) -> IngestReport:
        """Process one day end to end."""
        report = IngestReport(date=date)
        transcript = self._format(date, conversations)

        extracted = self._extract(date, transcript, report)
        report.extracted = len(extracted)

        if extracted:
            consolidated = self._consolidate(extracted, report)
            fresh = self._dedupe(consolidated, report)
            self._write(fresh, date, speaker, report)

        report.chunks_indexed = self._index(date, conversations, speaker)
        return report

    # ── [1] Extract ─────────────────────────────────────────────────

    def _extract(self, date: str, transcript: str, report: IngestReport) -> List[Dict]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACT_SYSTEM),
            ("human",
             "ALREADY RECORDED — reuse these spellings when they fit:\n{known}\n\n"
             "CONVERSATION DATE: {date}\n\n"
             "TRANSCRIPT:\n{transcript}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "known": self._known_block(),
                "date": date,
                "transcript": transcript,
            })
        except Exception as e:
            report.errors.append(f"extract failed: {e}")
            return []

        if not isinstance(result, dict):
            return []

        items = []
        for memory_type in ("preferences", "facts", "events"):
            type_label = memory_type.rstrip("s")  # preference, fact, event
            for i in result.get(memory_type, []):
                if not isinstance(i, dict) or not i.get("entity") or not i.get("content"):
                    continue
                items.append({
                    "entity": str(i["entity"]).lower().strip(),
                    "content": str(i["content"]).strip(),
                    "date": str(i.get("date") or date).strip(),
                    "type": type_label,
                    "speaker": str(i["speaker"]).lower().strip() if i.get("speaker") else None,
                })
        return items

    def _known_block(self) -> str:
        """
        The existing taxonomy, each entity followed by the values recorded under it,
        grouped by type (preferences, facts, events).

        Showing entity names alone keeps categories consistent but lets the values
        drift — the same choice gets written as "yoga" one day and "morning yoga" the
        next, and a question about when one changed to the other then finds a
        transition that never happened. Listing the values is what makes them reusable.

        Values are capped per entity: a settled entity has a handful, and one that has
        sprouted dozens is better served by showing the common ones than by spending
        the context on its long tail.
        """
        by_type: Dict[str, Dict[str, List[str]]] = {}
        for row in self.store.query_memories(limit=VALUE_SCAN_LIMIT):
            mem_type = row.get("type", "preference")
            grouped = by_type.setdefault(mem_type, {})
            grouped.setdefault(row["entity"], [])
            if row["content"] not in grouped[row["entity"]]:
                grouped[row["entity"]].append(row["content"])

        if not by_type:
            return "(nothing recorded yet — you are choosing the first names)"

        sections = []
        for type_label in ("preference", "fact", "event"):
            grouped = by_type.get(type_label, {})
            if not grouped:
                continue
            lines = [f"[{type_label}s]"]
            for entity, values in sorted(grouped.items()):
                lines.append(f"  - {entity}: {', '.join(values[:MAX_VALUES_SHOWN])}")
            sections.append("\n".join(lines))

        return "\n".join(sections)

    # ── [1.5] Consolidate ────────────────────────────────────────────

    def _consolidate(self, extracted: List[Dict], report: IngestReport) -> List[Dict]:
        """
        Normalize near-duplicate values to existing canonical spellings.

        For each extracted record, if the entity already exists in the store,
        compare the new value against all known values for that entity. If a
        near-match is found (token Jaccard > threshold), replace the new value
        with the existing one. This ensures "oat milk latte" and "oat latte"
        don't look like two different choices.

        Runs before dedup, so a normalized value that now exactly matches an
        existing (entity, content, date) row is caught by the dedup step.
        """
        known_values = self._values_by_entity()
        if not known_values:
            return extracted

        result = []
        for item in extracted:
            entity = item["entity"]
            if entity not in known_values:
                result.append(item)
                continue

            content = item["content"]
            best_match, best_score = None, 0.0
            for existing_val in known_values[entity]:
                if content.lower() == existing_val.lower():
                    best_match = existing_val
                    best_score = 1.0
                    break
                score = _token_jaccard(content, existing_val)
                if score > best_score:
                    best_match = existing_val
                    best_score = score

            if best_score >= CONSOLIDATION_THRESHOLD and best_match and content != best_match:
                item = {**item, "content": best_match}
                report.consolidated += 1

            result.append(item)

        return result

    def _values_by_entity(self) -> Dict[str, List[str]]:
        """All distinct values per entity, for consolidation lookups."""
        grouped: Dict[str, List[str]] = {}
        for row in self.store.query_memories(limit=VALUE_SCAN_LIMIT):
            entity = row["entity"]
            if entity not in grouped:
                grouped[entity] = []
            if row["content"] not in grouped[entity]:
                grouped[entity].append(row["content"])
        return grouped

    # ── [2] Dedupe ──────────────────────────────────────────────────

    def _dedupe(self, extracted: List[Dict], report: IngestReport) -> List[Dict]:
        """
        Drop exact repeats of (entity, content, date), within the batch and against
        what is stored.

        Only an identical record counts. The same value on a different date is a
        separate observation and is always kept — frequency is what makes a cycle
        visible, so collapsing recurrences would erase the pattern. Two people can
        also hold the same value on the same day, so the speaker is part of the key.
        """
        seen = set()
        fresh = []

        for item in extracted:
            key = (item["entity"], item["content"].lower(), item["date"],
                   item.get("speaker"))
            if key in seen or self._already_stored(item):
                report.duplicates += 1
                continue
            seen.add(key)
            fresh.append(item)

        return fresh

    def _already_stored(self, item: Dict) -> bool:
        rows = self.store.query_memories(
            entity=item["entity"],
            speaker=item.get("speaker"),
            start_date=item["date"],
            end_date=item["date"],
            limit=MAX_SAME_DAY_ROWS,
        )
        target = item["content"].lower()
        return any(r["content"].lower() == target for r in rows)

    # ── [3] Write ───────────────────────────────────────────────────

    def _write(
        self, items: List[Dict], date: str, speaker: str, report: IngestReport
    ) -> None:
        if not items:
            return

        self.store.add_memories(
            [
                {
                    "entity": i["entity"],
                    "content": i["content"],
                    "date": i["date"],
                    "type": i.get("type", "preference"),
                    "speaker": i.get("speaker") or speaker,
                }
                for i in items
            ],
            source_date=date,
        )
        report.added = len(items)
        report.entities = sorted({i["entity"] for i in items})
        from collections import Counter
        report.type_counts = dict(Counter(i.get("type", "preference") for i in items))

    # ── [4] Index ───────────────────────────────────────────────────

    def _index(self, date: str, conversations: List[Dict], speaker: str) -> int:
        count = 0
        for text, idx in self._chunks(date, conversations):
            self.store.add_conversation(
                text=text,
                metadata={"source_date": date, "chunk": idx, "speaker": speaker},
            )
            count += 1
        return count

    def _chunks(self, date: str, conversations: List[Dict]):
        turns = [
            (c.get("time_of_day", ""), t["speaker"], t["text"])
            for c in conversations
            for t in c.get("turns", [])
        ]
        for idx, start in enumerate(range(0, len(turns), CHUNK_SIZE)):
            batch = turns[start:start + CHUNK_SIZE]
            if not batch:
                continue
            lines, current = [f"Date: {date}"], None
            for time, sp, text in batch:
                if time and time != current:
                    lines.append(f"[{time}]")
                    current = time
                lines.append(f"{sp}: {text}")
            yield "\n".join(lines), idx

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format(date: str, conversations: List[Dict]) -> str:
        lines = [f"Date: {date}"]
        for conv in conversations:
            time = conv.get("time_of_day", "")
            if time:
                lines.append(f"\n[{time}]")
            for turn in conv.get("turns", []):
                lines.append(f"{turn['speaker']}: {turn['text']}")
        return "\n".join(lines)
