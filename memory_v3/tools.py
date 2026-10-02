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
    def semantic_retrieve_memory(query: str) -> str:
        """Multi-resolution memory retrieval — searches every layer of the hierarchy
        independently and returns results from each.

        CALL THIS FIRST for every question. It returns:
        - DISTILLED: focused knowledge documents tracking preferences, facts, and
          events with temporal evolution and reasoning
        - LIFETIME: the overall profile and confirmed patterns
        - YEARLY: major events and validated habits for each year
        - MONTHLY: consolidated facts and pattern details
        - WEEKLY: specific events and transition details

        This gives you both the big picture and the relevant details in one call.
        After reading the result, either answer directly or drill down with the
        other tools if you need exact dates, specific wording, or structured queries.

        Args:
            query: What to search for, as a topic or phrase.
        """
        layer_results = store.search_summaries_by_layer(
            query, k_week=2, k_month=2, k_year=1, k_lifetime=1
        )
        if not layer_results:
            return "(no summaries available — use search_memories or semantic_search_conversations instead)"

        sections = []
        for level in ("distilled", "lifetime", "year", "month", "week"):
            docs = layer_results.get(level, [])
            if not docs:
                continue
            header = level.upper()
            body = "\n\n".join(d.page_content for d in docs)
            sections.append(f"═══ {header} ═══\n{body}")
        return "\n\n".join(sections)

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

        Use after semantic_retrieve_memory when you need precise dates, ordering, counting,
        transitions, or specific facts/events. Results come back in chronological order.

        The store holds three types:
        - "preference": recurring choices within a category.
        - "fact": stable attributes about the person.
        - "event": one-time occurrences or milestones.

        Naming an entity that does not exist returns the list of real ones.

        Args:
            entity: Restrict to a single entity/category. Omit to search all.
            speaker: Restrict to one recorded speaker. Omit for all.
            start_date: Earliest date to include, YYYY-MM-DD.
            end_date: Latest date to include, YYYY-MM-DD.
            type: Restrict to one memory type. Omit for all.
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
                    "Omit `speaker` to search all."
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

        Use after semantic_retrieve_memory when you need the original wording of a discussion.
        Matches on meaning, not dates — use read_conversations_on for date lookups.

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

        Use when a question names a specific day. Embeddings carry meaning, not
        calendars, so date lookups must go through this tool.

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
    def list_entities() -> str:
        """List every entity and speaker the store holds.

        Use when you need the exact entity names for search_memories queries.
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

    return [
        semantic_retrieve_memory,
        search_memories,
        semantic_search_conversations,
        read_conversations_on,
        list_entities,
    ]
