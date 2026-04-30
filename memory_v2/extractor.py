"""
LLM-based memory extractor.

Takes raw conversation turns for a given date and returns a flat list of
memories. The LLM decides what type and subject each memory belongs to —
no hardcoded enums. Results are deduplicated against existing memories
before being written to the store.
"""

import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .store import MemoryStore

EXTRACTION_SYSTEM = """\
You are a memory extraction agent. Extract every piece of information worth remembering long-term from the conversation.

Each memory has four fields:
  "type"    : preference | fact | goal | event | routine | relationship | opinion (or anything fitting)
  "subject" : the main entity the memory is about (e.g. "coffee", "shoes", "exercise")
  "speaker" : who this memory belongs to — use their actual name if visible (e.g. "caroline", "melanie").
              For single-person conversations use "user".
  "content" :
    - preference → bare value only, noun/noun-phrase, no verbs, no speaker name, no time words
                   e.g. subject "coffee", speaker "user" → content "espresso"
    - all others → one complete sentence naming the speaker
                   e.g. "Caroline wants to finish her counseling degree by next year."

TEMPORAL RESOLUTION — critical:
The conversation starts with "Date: YYYY-MM-DD". Use this date to resolve ALL relative
time references into absolute dates before writing the content.
  "yesterday"        → session_date - 1 day  → write the resolved date
  "last Saturday"    → compute the actual date → write it
  "a few days ago"   → approximate (session_date - 3 days) → write the resolved date
  "last week"        → session_date - 7 days → write the resolved date
  "next month"       → keep as approximate future date

Always include the resolved absolute date in the content for events and facts with timing.
e.g. session date 2023-05-08, "I went to a support group yesterday"
  → content: "Caroline attended an LGBTQ support group on 2023-05-07."

Skip small talk. Return ONLY: {{"memories": [...]}}
"""

DEDUP_SYSTEM = """\
You are a memory deduplication agent.

You will receive:
  1. EXISTING memories already stored for this subject.
  2. NEW memories just extracted from today's conversations.

Return only the new memories that add information not already captured.
A memory is a duplicate if it says essentially the same thing as an existing one.
A memory is an UPDATE if it contradicts an existing one — keep it.

CONTENT FORMAT — return each memory exactly as received in NEW. Do not rephrase.
  preference → content stays as a bare value (e.g. "espresso", "blue t-shirt")
  others     → content stays as the original sentence

Return ONLY a JSON object: {{"memories": [...]}}
Each item keeps the same {{type, subject, content}} structure.
"""


class MemoryExtractor:
    def __init__(self, store: MemoryStore, model: str = "gpt-4o-mini"):
        self.store = store
        self.llm = ChatOpenAI(model=model, temperature=0)

    def extract_and_store(self, date: str, conversations: List[Dict]) -> List[Dict]:
        """
        Main entry point. Extracts memories from a day's conversations,
        deduplicates against existing, and writes to the store.

        conversations: list of {time_of_day, turns: [{speaker, text}]}
        Returns the new memories that were actually stored.
        """
        raw_text = self._format_conversations(date, conversations)
        extracted = self._extract(raw_text)

        if not extracted:
            return []

        new_memories = self._deduplicate(extracted)

        if new_memories:
            self.store.add_memories(new_memories, source_date=date)
            print(f"  [{date}] Stored {len(new_memories)} new memories "
                  f"(extracted {len(extracted)}, deduped {len(extracted)-len(new_memories)})")

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

    def _extract(self, raw_text: str) -> List[Dict]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM),
            ("human", "{text}"),
        ])
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"text": raw_text})
            if result is None:
                return []
            memories = result.get("memories", result) if isinstance(result, dict) else result
            if not isinstance(memories, list):
                return []
            return [m for m in memories if isinstance(m, dict) and m.get("content")]
        except Exception as e:
            print(f"  Extraction error: {e}")
            return []

    def _deduplicate(self, extracted: List[Dict]) -> List[Dict]:
        """Group by subject and deduplicate each group against existing DB entries."""
        from collections import defaultdict
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for m in extracted:
            groups[m.get("subject", "")].append(m)

        survivors = []
        for subject, new_batch in groups.items():
            existing = self.store.query_memories(subject=subject, limit=50)
            if not existing:
                survivors.extend(new_batch)
                continue

            existing_text = "\n".join(
                f"- [{e['type']}] {e['content']}" for e in existing
            )
            new_text = json.dumps(new_batch, ensure_ascii=False)

            prompt = ChatPromptTemplate.from_messages([
                ("system", DEDUP_SYSTEM),
                ("human",
                 "EXISTING:\n{existing}\n\nNEW:\n{new}\n\n"
                 "Return only the non-duplicate new memories."),
            ])
            try:
                chain = prompt | self.llm | JsonOutputParser()
                result = chain.invoke({"existing": existing_text, "new": new_text})
                kept = result.get("memories", result) if isinstance(result, dict) else result
                if isinstance(kept, list):
                    survivors.extend(kept)
                else:
                    survivors.extend(new_batch)
            except Exception as e:
                print(f"  Dedup error for subject '{subject}': {e}. Keeping all.")
                survivors.extend(new_batch)

        return survivors
