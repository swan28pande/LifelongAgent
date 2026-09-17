"""
LLM-based memory extractor.

Takes raw conversation turns for a given date and returns a flat list of
preference memories. The LLM decides what entity each memory belongs to —
no hardcoded enums. Results are deduplicated against existing memories
before being written to the store.
"""

import json
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .store import MemoryStore

EXTRACTION_SYSTEM = """\
You are a preference extraction agent. Extract only user preferences and daily routine choices from the conversation.

A preference or daily choice is a selection or liking for a specific value within a category (entity).
Each preference memory has four fields:
  "entity"  : the main category the preference/choice belongs to.
  "speaker" : who this preference/choice belongs to (usually "user")
  "content" : the CORE choice/value only (noun/noun-phrase).
  "date"    : the date this preference/choice applies to (format YYYY-MM-DD). If the choice/activity is explicitly performed or scheduled for a day other than today (e.g. "tomorrow"), calculate and use that specific target date. Otherwise, use the conversation date.

DAILY ROUTINES & CHOICES:
- A preference or daily choice includes any daily routine selections, options, statuses, or recovery/rest states the user prefers or performs.
- You MUST extract these choices for the specific day they are performed or selected.
- If the user explicitly mentions a recovery/rest state, status, or taking a break, extract it as a choice (e.g. content "rest day" or "no work").
- If the user is just discussing a hobby in general (e.g., "I like rock climbing") or planning a future activity (e.g., "I might climb next week"), do NOT extract it as a daily routine choice or workout status for today's date. Keep daily routine/exercise tracking strictly to what they actually did or chose on that specific date.
- Avoid extracting redundant constituent parts, ingredients, or reasons for a choice as separate preferences. For example, if the user chooses or prefers "pepperoni pizza", extract only "pepperoni pizza", and do NOT extract a separate preference for "pepperoni" unless they explicitly declare a separate preference for it independent of pizza (e.g. "I only buy pepperoni slices for snacks").
- ONLY extract choices, preferences, or selections performed, declared, or scheduled for a specific date. Do not extract generic future plans or hypothetical statements.

EXAMPLES:
- entity "operating system", content "macOS", date "2026-03-01"
- entity "programming language", content "python", date "2026-03-01"
- entity "music", content "jazz", date "2026-03-01"
- entity "cuisine", content "japanese", date "2026-03-01"
- entity "pet", content "golden retriever", date "2026-03-01"
- entity "work shift", content "night shift", date "2026-03-01"

REPETITION:
- Extract a preference even if it is already in the "EXISTING PREFERENCES" list.
- Record every mention to track consistency and frequency.

ENTITY CONSISTENCY:
- Look at the "EXISTING PREFERENCES" list.
- Use the exact same "entity" and "content" string if the user is referring to a choice they have made before.

TEMPORAL RESOLUTION:
- The conversation starts with "Date: YYYY-MM-DD". Use this to resolve relative time references (e.g. "tomorrow", "yesterday", "next week") to their absolute YYYY-MM-DD dates.
- If the user refers to tomorrow's activity, calculate tomorrow's date using the conversation date and put it in the "date" field.

Return ONLY a JSON object: {{"memories": [...]}}
"""

# DEDUP_SYSTEM = """\

# You are a memory deduplication agent.

# You will receive:
#   1. EXISTING memories already stored for this entity.
#   2. NEW memories just extracted from today's conversations.

# Return only the new memories that add information not already captured.
# A memory is a duplicate if it says essentially the same thing as an existing one.
# A memory is an UPDATE if it contradicts an existing one — keep it.

# CONTENT FORMAT — return each memory exactly as received in NEW. Do not rephrase.
#   preference → content stays as a bare value (e.g. "espresso", "blue t-shirt")
#   others     → content stays as the original sentence

# Return ONLY a JSON object: {{"memories": [...]}}
# Each item keeps the same {{type, entity, content}} structure.
# """


class MemoryExtractor:
    def __init__(self, store: MemoryStore, model: str = "gpt-5.5", llm=None):
        self.store = store
        if llm is not None:
            self.llm = llm
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, temperature=0)

    def extract_and_store(self, date: str, conversations: List[Dict]) -> List[Dict]:
        """
        Main entry point. Extracts memories from a day's conversations,
        deduplicates against existing, and writes to the store.

        conversations: list of {time_of_day, turns: [{speaker, text}]}
        Returns the new memories that were actually stored.
        """
        raw_text = self._format_conversations(date, conversations)
        existing_entities = self.store.get_all_entities()
        # Fetch unique existing preferences to maintain consistency
        unique_prefs = self.store.get_unique_preferences()
        existing_prefs = [f"[{m['entity']}] {m['content']}" for m in unique_prefs]

        extracted = self.extract_memories(raw_text, date, existing_entities, existing_prefs)

        if not extracted:
            return []

        # Deduplicate only against memories from the SAME day to reduce noise while keeping daily tracking
        # new_memories = self._deduplicate(extracted, date_filter=date)

        new_memories = extracted

        if new_memories:
            self.store.add_memories(new_memories, source_date=date)
            print(f"  [{date}] Stored {len(new_memories)} new memories (extracted {len(extracted)})")

        for chunk_text, chunk_idx in self._chunk_conversations(date, conversations):
            self.store.add_conversation(
                text=chunk_text,
                metadata={"source_date": date, "chunk": chunk_idx},
            )

        return new_memories

    # ── Private helpers ─────────────────────────────────────────────

    CHUNK_SIZE = 5  # turns per FAISS document


    def _format_conversations(self, date: str, conversations: List[Dict]) -> str:
        """Full session text used for memory extraction."""
        lines = [f"Date: {date}"]
        for conv in conversations:
            time = conv.get("time_of_day", "")
            if time:
                lines.append(f"\n[{time}]")
            for turn in conv.get("turns", []):
                lines.append(f"{turn['speaker']}: {turn['text']}")
        return "\n".join(lines)

    def _chunk_conversations(self, date: str, conversations: List[Dict]):
        """
        Split all turns into fixed-size chunks of CHUNK_SIZE turns each.
        Each chunk is stored as a separate FAISS document for precise retrieval.
        Yields (chunk_text, chunk_index).
        """
        # Collect all turns across all time-blocks for the day
        all_turns = []
        for conv in conversations:
            time = conv.get("time_of_day", "")
            for turn in conv.get("turns", []):
                all_turns.append((time, turn["speaker"], turn["text"]))

        chunk_idx = 0
        for start in range(0, max(len(all_turns), 1), self.CHUNK_SIZE):
            batch = all_turns[start: start + self.CHUNK_SIZE]
            if not batch:
                continue
            lines = [f"Date: {date}"]
            current_time = None
            for time, speaker, text in batch:
                if time and time != current_time:
                    lines.append(f"[{time}]")
                    current_time = time
                lines.append(f"{speaker}: {text}")
            yield "\n".join(lines), chunk_idx
            chunk_idx += 1

    def extract_memories(self, text: str, date: str, existing_entities: List[str] = [],
                         existing_prefs: List[str] = []) -> List[Dict]:
        """Extract memories from a conversation text."""
        try:
            entities_str = "\n".join(f"  - {s}" for s in existing_entities) if existing_entities else "  (None yet)"
            prefs_str = "\n".join(f"  - {p}" for p in set(existing_prefs)) if existing_prefs else "  (None yet)"

            prompt = ChatPromptTemplate.from_messages([
                ("system", EXTRACTION_SYSTEM),
                ("user", "Date: {date}\n\nExisting Entities:\n{existing_entities}\n\nExisting Preferences (for consistency):\n{existing_prefs}\n\nConversation:\n{text}")
            ])
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "text": text,
                "date": date,
                "existing_entities": entities_str,
                "existing_prefs": prefs_str
            })
            if result is None:
                return []
            memories = result.get("memories", result) if isinstance(result, dict) else result
            if not isinstance(memories, list):
                return []
            return [m for m in memories if isinstance(m, dict) and m.get("content")]
        except Exception as e:
            print(f"  Extraction error: {e}")
            return []

    # def _deduplicate(self, extracted: List[Dict], date_filter: Optional[str] = None) -> List[Dict]:
    #     """Group by entity and deduplicate each group against existing DB entries."""
    #     from collections import defaultdict
    #     groups: Dict[str, List[Dict]] = defaultdict(list)
    #     for m in extracted:
    #         groups[m.get("entity", "")].append(m)

    #     survivors = []
    #     for entity, new_batch in groups.items():
    #         existing = self.store.query_memories(
    #             entity=entity,
    #             start_date=date_filter,
    #             end_date=date_filter,
    #             limit=50
    #         )
    #         if not existing:
    #             survivors.extend(new_batch)
    #             continue

    #         existing_text = "\n".join(
    #             f"- [{e['type']}] {e['content']}" for e in existing
    #         )
    #         new_text = json.dumps(new_batch, ensure_ascii=False)

    #         prompt = ChatPromptTemplate.from_messages([
    #             ("system", DEDUP_SYSTEM),
    #             ("human",
    #              "EXISTING:\n{existing}\n\nNEW:\n{new}\n\n"
    #              "Return only the non-duplicate new memories."),
    #         ])
    #         try:
    #             chain = prompt | self.llm | JsonOutputParser()
    #             result = chain.invoke({"existing": existing_text, "new": new_text})
    #             kept = result.get("memories", result) if isinstance(result, dict) else result
    #             if isinstance(kept, list):
    #                 survivors.extend(kept)
    #             else:
    #                 survivors.extend(new_batch)
    #         except Exception as e:
    #             print(f"  Dedup error for entity '{entity}': {e}. Keeping all.")
    #             survivors.extend(new_batch)

    #     return survivors
