"""
Ingestion pipeline.

    day's turns
        │
        ▼
    [1] extract      one LLM call  → dated (entity, content) records
        │
        ▼
    [2] dedupe       deterministic → drop exact repeats of (entity, content, date)
        │
        ▼
    [3] write        deterministic → SQL inserts
        │
        ▼
    [4] index        deterministic → fixed-size chunks into the RAG store

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


@dataclass
class IngestReport:
    """What the pipeline did with one day. Returned so runs can be audited."""

    date: str
    extracted: int = 0
    added: int = 0
    duplicates: int = 0
    chunks_indexed: int = 0
    entities: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"[{self.date}] extracted {self.extracted}", f"added {self.added}"]
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
            fresh = self._dedupe(extracted, report)
            self._write(fresh, date, speaker, report)

        report.chunks_indexed = self._index(date, conversations, speaker)
        return report

    # ── [1] Extract ─────────────────────────────────────────────────

    def _extract(self, date: str, transcript: str, report: IngestReport) -> List[Dict]:
        known = self.store.get_all_entities()
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACT_SYSTEM),
            ("human",
             "KNOWN ENTITIES (reuse these names when they fit):\n{known}\n\n"
             "CONVERSATION DATE: {date}\n\n"
             "TRANSCRIPT:\n{transcript}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "known": "\n".join(f"- {e}" for e in known) or "(none yet)",
                "date": date,
                "transcript": transcript,
            })
        except Exception as e:
            report.errors.append(f"extract failed: {e}")
            return []

        items = result.get("preferences", []) if isinstance(result, dict) else []
        return [
            {
                "entity": str(i["entity"]).lower().strip(),
                "content": str(i["content"]).strip(),
                "date": str(i.get("date") or date).strip(),
            }
            for i in items
            if isinstance(i, dict) and i.get("entity") and i.get("content")
        ]

    # ── [2] Dedupe ──────────────────────────────────────────────────

    def _dedupe(self, extracted: List[Dict], report: IngestReport) -> List[Dict]:
        """
        Drop exact repeats of (entity, content, date), within the batch and against
        what is stored.

        Only an identical triple counts. The same value on a different date is a
        separate observation and is always kept — frequency is what makes a cycle
        visible, so collapsing recurrences would erase the pattern.
        """
        seen = set()
        fresh = []

        for item in extracted:
            key = (item["entity"], item["content"].lower(), item["date"])
            if key in seen or self._already_stored(item):
                report.duplicates += 1
                continue
            seen.add(key)
            fresh.append(item)

        return fresh

    def _already_stored(self, item: Dict) -> bool:
        rows = self.store.query_memories(
            entity=item["entity"],
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
                    "speaker": speaker,
                }
                for i in items
            ],
            source_date=date,
        )
        report.added = len(items)
        report.entities = sorted({i["entity"] for i in items})

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
