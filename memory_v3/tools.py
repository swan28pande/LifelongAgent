"""
Retrieval tools for the chat agent.

Built by a factory that closes over the MemoryStore and context builder, so the agent
sees only the tool surface and never the objects behind it.

These are read-only by construction. Writing happens in the ingestion pipeline, which
calls the store directly rather than through tools.

Signatures are deliberately flat (no nested object arguments): Gemini and OpenAI
function-calling both handle scalar parameters reliably, and the agent can issue
several calls in parallel instead of batching into one nested payload.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from langchain_core.tools import tool

from .store import MemoryStore

MAX_ROWS = 200

# Cap on how far `read_conversations_on` will widen around a date. A window this size
# already spans a full week either way; anything broader is a search, not a lookup.
MAX_WINDOW_DAYS = 7


def _fmt_rows(rows: List[dict]) -> str:
    if not rows:
        return "(no matching memories)"
    return "\n".join(
        f"id={r['id']} [{r['date']}] [{r.get('type', 'preference')}] ({r['entity']}) {r['content']} — {r['speaker']}"
        for r in rows
    )



def build_read_tools(store: MemoryStore) -> list:
    """Retrieval tools for the chat agent."""

    @tool
    def list_entities() -> str:
        """List what the store holds: every entity, and every speaker on record.

        Call this first. It maps a vaguely worded question onto the entity name
        actually used in storage, and it tells you whether filtering by speaker is
        meaningful — with one speaker it never is, with several it is essential.
        """
        entities = store.get_all_entities()
        speakers = store.get_all_speakers()
        if not entities and not speakers:
            return "(store is empty)"

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

    @tool
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

        Most stores hold a single speaker. Omit `speaker` unless you have confirmed
        from `list_entities` or an earlier result that several people are recorded —
        filtering on the name of the person the question is about will usually match
        nothing, because that is not how the speaker field is filled in.

        Args:
            entity: Restrict to a single entity/category. Omit to search all.
            speaker: Restrict to one recorded speaker. Omit for all.
            start_date: Earliest date to include, YYYY-MM-DD.
            end_date: Latest date to include, YYYY-MM-DD.
            type: Restrict to one memory type: "preference", "fact", or "event". Omit for all.
            limit: Maximum rows to return.
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
                    "Omit `speaker` to search all — the person a question is about is "
                    "usually not stored as a separate speaker."
                )

        rows = store.query_memories(
            entity=entity,
            speaker=speaker,
            start_date=start_date,
            end_date=end_date,
            type=type,
            limit=min(limit, MAX_ROWS),
        )
        return _fmt_rows(rows)

    @tool
    def semantic_search_conversations(query: str, k: int = 5) -> str:
        """Semantic search over raw conversation chunks, by meaning.

        Finds exchanges about a topic. It matches on wording, not on when something
        was said, so searching for a date here will not reliably find that day — use
        `read_conversations_on` when you know the date you want.

        Args:
            query: What to search for, as a topic or phrase.
            k: Number of chunks to return.
        """
        docs = store.search_conversations(query, k=k)
        if not docs:
            return "(no matching conversations)"
        return "\n\n---\n\n".join(d.page_content for d in docs)

    @tool
    def read_conversations_on(date: str, days_around: int = 0) -> str:
        """Read the conversation from a specific date, exactly as it was said.

        The right tool whenever a question names a day. Semantic search cannot find a
        date reliably — embeddings carry meaning, not calendars — so looking a date up
        by keyword tends to return the wrong days or nothing at all.

        Widen `days_around` when a day mentions something that happened earlier: a
        choice made on one day is often only described the morning after.

        Args:
            date: The day to read, YYYY-MM-DD.
            days_around: Also include this many days either side. 0 reads one day.
        """
        try:
            centre = datetime.strptime(date.strip(), "%Y-%m-%d")
        except ValueError:
            return f"'{date}' is not a date. Use YYYY-MM-DD."

        window = max(0, min(days_around, MAX_WINDOW_DAYS))
        start = (centre - timedelta(days=window)).strftime("%Y-%m-%d")
        end = (centre + timedelta(days=window)).strftime("%Y-%m-%d")

        docs = store.get_conversations_by_date_range(start, end)
        if not docs:
            first, last = store.get_date_range()
            span = f"{first} to {last}" if first else "(store is empty)"
            return (
                f"Nothing recorded for {start}"
                + (f" to {end}" if window else "")
                + f". Recorded range is {span}."
            )

        return "\n\n---\n\n".join(d.page_content for d in docs)

    @tool
    def get_summary(level: str, identifier: str = "", speaker: str = "user") -> str:
        """Fetch one stored summary by level.

        Args:
            level: One of "week", "month", "year", "lifetime".
            identifier: The period id — "2026-W10" for week, "2026-03" for month,
                "2026" for year. Leave empty for lifetime.
            speaker: Whose summary to fetch. Defaults to "user".
        """
        level = level.lower().strip()
        if level == "lifetime":
            text = store.get_lifetime_summary(speaker)
        else:
            text = store.get_summary(f"{level}:{identifier}:{speaker}")
        return text if text else f"(no {level} summary found for {identifier or speaker})"

    @tool
    def semantic_search_summaries(query: str, k: int = 3) -> str:
        """Semantic search across all stored summaries — weekly, monthly, yearly, lifetime.

        Use this when the question is about someone's background, life events, general
        habits, or long-term patterns and you don't know which specific time period to
        look at. It searches by meaning across every summary in the store.

        Prefer this over search_conversations for broad "who/what/why" questions — summaries
        are pre-digested overviews that cover more ground per result.

        Args:
            query: What to search for, as a topic or phrase.
            k: Number of summaries to return.
        """
        docs = store.search_summaries(query, k=k)
        if not docs:
            return "(no matching summaries)"
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return [
        list_entities,
        search_memories,
        semantic_search_conversations,
        semantic_search_summaries,
        read_conversations_on,
        get_summary,
    ]
