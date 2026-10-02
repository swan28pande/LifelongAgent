"""
Retrieval tools for the chat agent.

Built by a factory that closes over the MemoryStore, so the agent
sees only the tool surface and never the objects behind it.

These are read-only by construction. Writing happens in the ingestion pipeline, which
calls the store directly rather than through tools.

Signatures are flat and arguments are validated before commitment. The reader
executes one selected tool at a time and reassesses after its result.
"""

from __future__ import annotations

import re
from datetime import date as calendar_date, timedelta
from typing import TYPE_CHECKING, Annotated, Any, List, Literal, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)

if TYPE_CHECKING:
    from .store import MemoryStore

MAX_ROWS = 200
MAX_SEARCH_RESULTS = 20

# Maximum days to include on each side of an exact date lookup.
MAX_WINDOW_DAYS = 7


def _normalize_name(value: Any) -> Any:
    return value.lower().strip() if isinstance(value, str) else value


def _date_argument(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("Use YYYY-MM-DD")
    calendar_date.fromisoformat(value)
    return value


def _nonempty_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Use nonempty text")
    return value


NameArgument = Annotated[
    str, BeforeValidator(_normalize_name), AfterValidator(_nonempty_text)
]
DateArgument = Annotated[str, AfterValidator(_date_argument)]
QueryArgument = Annotated[str, AfterValidator(_nonempty_text)]
RowLimit = Annotated[StrictInt, Field(ge=1, le=MAX_ROWS)]
SearchLimit = Annotated[StrictInt, Field(ge=1, le=MAX_SEARCH_RESULTS)]
DateWindow = Annotated[StrictInt, Field(ge=0, le=MAX_WINDOW_DAYS)]
MemoryType = Annotated[
    Literal["preference", "fact", "event"], BeforeValidator(_normalize_name)
]
SummaryLevel = Annotated[
    Literal["week", "month", "year", "lifetime"], BeforeValidator(_normalize_name)
]


class SearchMemoriesArguments(BaseModel):
    """Flat search arguments, including cross-field date validation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    entity: Optional[NameArgument] = None
    speaker: Optional[NameArgument] = None
    start_date: Optional[DateArgument] = None
    end_date: Optional[DateArgument] = None
    type: Optional[MemoryType] = None
    limit: RowLimit = 100

    @model_validator(mode="after")
    def ordered_dates(self) -> SearchMemoriesArguments:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class SummaryArguments(BaseModel):
    """Validate summary period identifiers before any retrieval runs."""

    model_config = ConfigDict(extra="forbid", strict=True)
    level: SummaryLevel
    identifier: str = ""
    speaker: NameArgument = "user"

    @model_validator(mode="after")
    def valid_period(self) -> SummaryArguments:
        self.identifier = self.identifier.strip()
        value = self.identifier
        if self.level == "lifetime":
            if value:
                raise ValueError("Lifetime summaries have no period identifier")
        elif self.level == "week" and re.fullmatch(r"[0-9]{4}-W[0-9]{2}", value):
            calendar_date.fromisocalendar(int(value[:4]), int(value[6:]), 1)
        elif self.level == "month" and re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
            calendar_date.fromisoformat(value + "-01")
        elif self.level == "year" and re.fullmatch(r"[0-9]{4}", value):
            calendar_date(int(value), 1, 1)
        else:
            raise ValueError(
                "Use YYYY-Www for week, YYYY-MM for month, or YYYY for year"
            )
        return self


def _fmt_rows(rows: List[dict]) -> str:
    if not rows:
        return "(no matching memories)"
    return "\n".join(
        f"id={r['id']} [{r['date']}] [{r.get('type', 'preference')}] ({r['entity']}) {r['content']} — {r['speaker']}"
        for r in rows
    )


def build_read_tools(store: MemoryStore) -> list[BaseTool]:
    """Retrieval tools for the chat agent."""

    @tool
    def semantic_retrieve_memory(query: QueryArgument) -> str:
        """Search each summary layer and return relevant context in one retrieval.

        Prefer this as the first lookup when stored memory is needed. Returns up to
        two distilled documents, one lifetime profile, one yearly summary, two
        monthly summaries, and two weekly summaries. Each level is selected
        independently, so broad profiles and specific details are both represented.

        Reassess the result before another lookup. Use get_summary for a known
        period, search_memories for exact facts/dates/counts, or raw conversations
        for original wording. Summaries and top-k matches are incomplete evidence.

        Args:
            query: The person, topic, or question to search for.
        """
        layer_results = store.search_summaries_by_layer(query.strip())
        if not layer_results:
            return "(no summaries available — use search_memories or semantic_search_conversations instead)"
        sections = []
        for level in ("distilled", "lifetime", "year", "month", "week"):
            docs = layer_results.get(level, [])
            if docs:
                body = "\n\n".join(doc.page_content for doc in docs)
                sections.append(f"═══ {level.upper()} ═══\n{body}")
        return "\n\n".join(sections)

    @tool
    def list_entities() -> str:
        """List what the store holds: every entity, and every speaker on record.

        Use this when the stored entity or speaker names are unknown. Skip it when
        the current context or an earlier result already supplies the names needed.
        With one speaker filtering is unnecessary; with several it can be essential.
        """
        entities = store.get_all_entities()
        speakers = store.get_all_speakers()
        if not entities and not speakers:
            return "(no structured memories; raw conversations or summaries may still exist)"

        if len(speakers) > 1:
            note = (
                f"SPEAKERS ({len(speakers)} on record — filter by these when a "
                f"question is about one of them): {', '.join(speakers)}"
            )
        else:
            note = (
                f"SPEAKERS: {', '.join(speakers) or 'none'} — only one on record, so "
                "do not filter by speaker."
            )

        return f"ENTITIES: {', '.join(entities) or 'none'}\n{note}"

    @tool(args_schema=SearchMemoriesArguments)
    def search_memories(
        entity: Optional[str] = None,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 100,
    ) -> str:
        """Query the structured memory database for an exact dated timeline.

        The precise source: use it whenever the answer depends on dates, ordering, or
        counting — transitions, cycles, what was chosen on a given day, what facts are
        known, or what events happened. Results come back in chronological order.

        The store holds three types of memory:
        - "preference": recurring choices within a category.
        - "fact": stable attributes about the person.
        - "event": one-time occurrences or milestones.

        Filter by type when you know what you need. Omit type to search across all.

        Naming an entity or speaker that does not exist returns the list of real ones
        rather than an empty result, so a wrong guess can be corrected on the next call.

        Memories are extracted for each named speaker. Use the recorded speaker
        name from context, summaries, or list_entities when asking about one person.
        Omit `speaker` when the question covers everyone.

        Args:
            entity: Restrict to a single entity/category. Omit to search all.
            speaker: Restrict to one recorded speaker. Omit for all.
            start_date: Earliest date to include, YYYY-MM-DD.
            end_date: Latest date to include, YYYY-MM-DD.
            type: Restrict to one memory type: "preference", "fact", or "event". Omit for all.
            limit: Maximum rows to return, from 1 to 200. Truncated results are marked.
        """
        if entity:
            known = store.get_all_entities()
            if entity.lower().strip() not in known:
                available = ", ".join(known) if known else "(none yet)"
                return (
                    f"No entity named '{entity}'. Available entities: {available}. "
                    "Retry with one of these, or omit `entity` to search all."
                )

        if speaker:
            speakers = store.get_all_speakers()
            if speaker.lower().strip() not in speakers:
                available = ", ".join(speakers) if speakers else "(none yet)"
                return (
                    f"No speaker named '{speaker}'. Recorded speakers: {available}. "
                    "Retry with a recorded name, or omit `speaker` to search all."
                )

        rows = store.query_memories(
            entity=entity,
            speaker=speaker,
            start_date=start_date,
            end_date=end_date,
            type=type,
            limit=limit + 1,
        )
        text = _fmt_rows(rows[:limit])
        if len(rows) > limit:
            text += f"\n[Results truncated to the earliest {limit} rows; narrow filters for complete coverage.]"
        return text

    @tool
    def semantic_search_conversations(query: QueryArgument, k: SearchLimit = 5) -> str:
        """Semantic search over raw conversation chunks, by meaning.

        Finds exchanges about a topic. It matches on wording, not on when something
        was said, so searching for a date here will not reliably find that day — use
        `read_conversations_on` when you know the date you want.

        Args:
            query: What to search for, as a topic or phrase.
            k: Number of chunks to return, from 1 to 20.
        """
        docs = store.search_conversations(query, k=k)
        if not docs:
            return "(no matching conversations)"
        return "\n\n---\n\n".join(d.page_content for d in docs)

    @tool
    def read_conversations_on(date: DateArgument, days_around: DateWindow = 0) -> str:
        """Read the conversation from a specific date, exactly as it was said.

        The right tool whenever a question names a day. Semantic search cannot find a
        date reliably — embeddings carry meaning, not calendars — so looking a date up
        by keyword tends to return the wrong days or nothing at all.

        Widen `days_around` when a day mentions something that happened earlier: a
        choice made on one day is often only described the morning after.

        Args:
            date: The day to read, YYYY-MM-DD.
            days_around: Include 0 to 7 days either side. 0 reads one day.
        """
        centre = calendar_date.fromisoformat(date.strip())
        start = (centre - timedelta(days=days_around)).isoformat()
        end = (centre + timedelta(days=days_around)).isoformat()

        docs = store.get_conversations_by_date_range(start, end)
        if not docs:
            first, last = store.get_date_range()
            span = f"{first} to {last}" if first else "(store is empty)"
            return (
                f"Nothing recorded for {start}"
                + (f" to {end}" if days_around else "")
                + f". Structured-memory date range is {span}; raw conversation coverage may differ."
            )

        return "\n\n---\n\n".join(d.page_content for d in docs)

    @tool(args_schema=SummaryArguments)
    def get_summary(level: str, identifier: str = "", speaker: str = "user") -> str:
        """Fetch one stored summary by level.

        Args:
            level: One of "week", "month", "year", "lifetime".
            identifier: The period id — "2026-W10" for week, "2026-03" for month,
                "2026" for year. Leave empty for lifetime.
            speaker: Whose summary to fetch. Defaults to "user".
        """
        # BaseTool validates scalar input but forwards its original text.
        level = level.lower().strip()
        if level == "lifetime":
            text = store.get_lifetime_summary(speaker)
        else:
            text = store.get_summary(f"{level}:{identifier}:{speaker}")
        return (
            text if text else f"(no {level} summary found for {identifier or speaker})"
        )

    return [
        semantic_retrieve_memory,
        list_entities,
        search_memories,
        semantic_search_conversations,
        read_conversations_on,
        get_summary,
    ]
