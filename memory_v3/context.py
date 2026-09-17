"""
Dynamic context builder.

The v2 ContextBuilder carries a hardcoded `mappings` table naming the domains it
expects to find, with hand-written keyword lists per domain. That only works on a
dataset whose categories are known in advance, and it silently returns nothing for any
domain not on the list.

This builder holds no domain knowledge. It reads the entity taxonomy out of the store
at call time and asks the LLM to match the query against whatever is actually there,
so it works on any dataset and degrades gracefully when the store is empty.
"""

from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from memory_v2.store import MemoryStore

ROUTER_SYSTEM = """\
Match a user's question against the memory categories that actually exist.

You are given the full list of entity names in the store. Choose the ones whose
recorded values could answer the question. Match on meaning, not spelling — the user's
wording will rarely equal the stored name.

Also decide whether answering needs a dated timeline of individual choices, or whether
a general profile is enough. Questions about what happened on a day, how often
something changes, what usually happens, or what comes next need the timeline.

Return ONLY JSON:
{{
  "entities": ["exact names copied from the provided list, or [] if none fit"],
  "needs_timeline": true or false,
  "speaker": "the person the question is about, or null",
  "target_date": "YYYY-MM-DD if one specific date is implied, else null"
}}

Copy entity names character-for-character from the list. Never invent one.
"""


class DynamicContextBuilder:
    def __init__(self, store: MemoryStore, llm):
        self.store = store
        self.llm = llm

    def build(self, query: str, n_conversations: int = 8, n_summaries: int = 3) -> str:
        entities = self.store.get_all_entities()
        route = self._route(query, entities)

        parts = []

        profile = self.store.get_lifetime_summary(route.get("speaker"))
        if profile:
            parts.append(f"[PROFILE]\n{self._strip_prefix(profile)}")

        if route.get("needs_timeline") and route.get("entities"):
            timeline = self._timeline(route["entities"], route.get("speaker"))
            if timeline:
                parts.append(timeline)

        prefs = self._preferences(route.get("entities", []), route.get("speaker"))
        if prefs:
            parts.append(f"[RELEVANT PREFERENCES]\n{prefs}")

        summaries = self.store.search_summaries(query, k=n_summaries)
        if summaries:
            parts.append(
                "[SUMMARIES]\n"
                + "\n\n---\n\n".join(d.page_content for d in summaries)
            )

        convs = self.store.search_conversations(query, k=n_conversations)
        if convs:
            parts.append(
                "[CONVERSATIONS]\n"
                + "\n\n---\n\n".join(d.page_content for d in convs)
            )

        return "\n\n".join(parts)

    # ── Internals ───────────────────────────────────────────────────

    def _route(self, query: str, entities: List[str]) -> dict:
        if not entities:
            return {"entities": [], "needs_timeline": False, "speaker": None}

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM),
            ("human", "AVAILABLE ENTITIES:\n{entities}\n\nQUESTION:\n{query}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "entities": "\n".join(f"- {e}" for e in entities),
                "query": query,
            })
        except Exception:
            return {"entities": [], "needs_timeline": True, "speaker": None}

        # Keep only names that really exist — the LLM may hallucinate or reword.
        known = {e.lower(): e for e in entities}
        matched = [
            known[e.lower()]
            for e in (result.get("entities") or [])
            if isinstance(e, str) and e.lower() in known
        ]

        speaker = result.get("speaker")
        if speaker:
            speaker = speaker.lower().strip()
            if speaker not in self.store.get_all_speakers():
                speaker = None

        return {
            "entities": matched,
            "needs_timeline": bool(result.get("needs_timeline")),
            "speaker": speaker,
            "target_date": result.get("target_date"),
        }

    def _timeline(self, entities: List[str], speaker: Optional[str]) -> str:
        blocks = []
        for entity in entities:
            rows = self.store.query_memories(
                entity=entity, speaker=speaker, limit=1000
            )
            if not rows:
                continue
            seen, lines = set(), []
            for r in sorted(rows, key=lambda x: x["date"]):
                key = (r["date"], r["content"].lower())
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {r['date']}: {r['content']}")
            if lines:
                blocks.append(f"[TIMELINE — {entity}]\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def _preferences(self, entities: List[str], speaker: Optional[str],
                     per_entity: int = 20) -> str:
        lines = []
        for entity in entities or [None]:
            for r in self.store.query_memories(
                entity=entity, speaker=speaker, limit=per_entity
            ):
                spk = f"[{r['speaker']}] " if r.get("speaker") else ""
                lines.append(f"{spk}[{r['date']}] ({r['entity']}) {r['content']}")
        return "\n".join(lines)

    @staticmethod
    def _strip_prefix(text: str) -> str:
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else text
