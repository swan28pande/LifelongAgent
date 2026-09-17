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

from typing import List, Optional

from langchain_core.tools import tool

from memory_v2.store import MemoryStore

from .context import DynamicContextBuilder

MAX_ROWS = 200


def _fmt_rows(rows: List[dict]) -> str:
    if not rows:
        return "(no matching preferences)"
    return "\n".join(
        f"id={r['id']} [{r['date']}] ({r['entity']}) {r['content']} — {r['speaker']}"
        for r in rows
    )



def build_read_tools(store: MemoryStore, ctx_builder: DynamicContextBuilder) -> list:
    """Tools for chat mode: assembled context plus targeted retrieval."""

    @tool
    def build_context(query: str) -> str:
        """Assemble a full memory context for a query in one call.

        Matches the query against the entities that exist in the store, then gathers
        the profile, relevant preferences, dated timelines, summaries, and matching
        conversations. This is the fast path — start here for most questions.

        Args:
            query: The user's question, passed through as-is.
        """
        ctx = ctx_builder.build(query)
        return ctx.strip() if ctx.strip() else "(no context found for this query)"

    @tool
    def list_entities() -> str:
        """List every preference entity (category) in the store.

        Use this to map a vaguely worded question onto the entity name actually used
        in storage before calling `search_preferences`.
        """
        entities = store.get_all_entities()
        return ", ".join(entities) if entities else "(store is empty — no entities yet)"

    @tool
    def search_preferences(
        entity: Optional[str] = None,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> str:
        """Query the structured preference database for an exact dated timeline.

        Use this over `build_context` when the answer depends on precise dates,
        ordering, or counting — transitions, cycles, "what did I choose on X".
        Results are ordered chronologically.

        Args:
            entity: Restrict to a single entity/category. Omit to search all.
            speaker: Restrict to one person. Omit for all.
            start_date: Earliest date to include, YYYY-MM-DD.
            end_date: Latest date to include, YYYY-MM-DD.
            limit: Maximum rows to return.
        """
        rows = store.query_memories(
            entity=entity,
            speaker=speaker,
            start_date=start_date,
            end_date=end_date,
            limit=min(limit, MAX_ROWS),
        )
        return _fmt_rows(rows)

    @tool
    def search_conversations(query: str, k: int = 5) -> str:
        """Semantic search over raw conversation chunks for verbatim exchanges.

        Use when the user asks what was actually said, or when a preference lookup
        misses context that only the original wording carries.

        Args:
            query: What to search for.
            k: Number of chunks to return.
        """
        docs = store.search_conversations(query, k=k)
        if not docs:
            return "(no matching conversations)"
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

    return [
        build_context,
        list_entities,
        search_preferences,
        search_conversations,
        get_summary,
    ]
