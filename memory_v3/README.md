# memory_v3: current memory system

Writing and reading are built differently:

- **Write** is a fixed pipeline, one LLM call per day:
  extract → consolidate → dedupe → write → index.
- **Read** is a tool-calling agent, because how many lookups a question needs isn't
  known in advance.

Storage (SQLite + FAISS) and the summary hierarchy come from `memory_v2/`.

| File | What it does |
|---|---|
| [agent.py](agent.py) | `AgenticMemoryAgent`, the entry point. `ingest()` / `ingest_range()` for batch replay; `converse()` + `flush()` for live use (today stays in context, written at day's end); `chat()` / `chat_with_trace()` to answer; `build_summaries()`. Also `_make_llm()`, which picks Gemini on Vertex or OpenAI. |
| [ingest.py](ingest.py) | The write pipeline. **Extract**: one LLM call gets dated (entity, content) preference records, shown the existing values so spellings are reused. **Consolidate**: near-duplicate values (token Jaccard + character similarity ≥ 0.75) are rewritten to the existing spelling. **Dedupe**: exact (entity, content, date, speaker) repeats are dropped. **Write**: SQL insert. **Index**: 5-turn conversation chunks into SQL + FAISS. |
| [tools.py](tools.py) | The read agent's tools: `list_entities`, `search_preferences` (SQL, exact dated timeline), `semantic_search_conversations`, `semantic_search_summaries`, `read_conversations_on` (exact date lookup), `get_summary` (one summary by level and period). |
| [prompts.py](prompts.py) | `EXTRACT_SYSTEM` (what counts as a preference record) and `CHAT_SYSTEM` (retrieval strategy and answer format). Deliberately domain-blind, so they never name a category. |

```python
from memory_v3.agent import AgenticMemoryAgent
agent = AgenticMemoryAgent(base_dir="results/synthetic/v3/store")
agent.ingest("2026-03-01", [{"time_of_day": "Morning", "turns": [...]}], update_summaries=True)
agent.chat("What was I drinking in early March?")
```

`update_summaries=True` rebuilds only the week → month → year → lifetime summaries whose
source data changed (hash-checked); `flush()` does this by default.
