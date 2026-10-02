# Agent structure in memory_v3

This describes the current code in memory_v3 and its imported memory_v2 store and summarizer. The central distinction is simple: **one tool-calling agent handles reading**; the coordinator calls fixed workflows for writing and summarization.

~~~text
AgenticMemoryAgent
  ├─ store         MemoryStore: SQLite records + FAISS indexes
  ├─ pipeline      IngestionPipeline: write one day's memory
  ├─ summarizer    Summarizer: week → month → year → lifetime
  ├─ read_tools    six read-only retrieval operations
  ├─ chat_agent    the one LangChain tool-calling agent
  └─ _pending      today's turns, held in process memory
~~~

## Construction and ownership

The [constructor](../../memory_v3/agent.py#L100) creates one shared MemoryStore, then three LLM wrappers: extraction at temperature 0, answering at 0.3, and summarization at 0.2. By default they use the same model name. These wrappers serve different steps; they are not three collaborating agents.

The constructor gives the answering model six tools and the static CHAT_SYSTEM prompt through LangChain's create_agent, naming the resulting agent memory_reader. It also initializes an empty _pending list for live conversation. There is no custom supervisor or separate planner in this code. ([Construction](../../memory_v3/agent.py#L112), [reader prompt](../../memory_v3/prompts.py#L95))

## Reader: where the agent makes decisions

For [chat(query)](../../memory_v3/agent.py#L180), the coordinator creates a fresh HumanMessage and invokes chat_agent. The model can issue tool calls, inspect returned text, continue retrieving, and finally answer. The call uses a graph recursion limit of 60 by default; this caps graph steps, not exactly 60 tool calls. Reusing the chat_agent object does not carry previous chat messages into the next chat call. ([Invocation](../../memory_v3/agent.py#L287))

The tools are built by [build_read_tools](../../memory_v3/tools.py#L39), which closes over the shared store. The model sees tool names, descriptions, arguments, and returned strings rather than a store object or direct SQL access.

| Tool | Evidence returned | Main use |
|---|---|---|
| list_entities | Stored entity and speaker names | Discover the store's vocabulary |
| search_memories | Chronological SQLite rows | Dates, changes, counts, facts, events |
| semantic_search_conversations | Similar raw chunks from FAISS | Find discussion by topic |
| read_conversations_on | Raw chunks from SQL for a date | Check exact wording or nearby days |
| semantic_search_summaries | Similar period summaries from FAISS | Broad profile or habit questions |
| get_summary | One identified period summary | Fetch a specific week, month, year, or lifetime profile |

All six are read-only. The [reader prompt](../../memory_v3/prompts.py#L95) suggests summaries for broad questions, structured rows for chronology, and raw conversation for details. Tool descriptions add more specific guidance. The model chooses the actual sequence; the application does not preselect a fixed set of retrievals.

[chat_with_trace](../../memory_v3/agent.py#L184) invokes the same agent but returns only its final answer plus each tool call's name and arguments. It does not include tool outputs or a complete message transcript.

## Writer and summarizer: caller-controlled workflows

[ingest(date, conversations)](../../memory_v3/agent.py#L129) calls [IngestionPipeline.run](../../memory_v3/ingest.py#L122). That pipeline formats one day's transcript, extracts typed memories with one LLM call, normalizes similar values, removes exact same-day repeats, writes surviving records to SQLite, and indexes five-turn raw conversation chunks in SQLite and FAISS. The reader has no write tool. The caller supplies the date and decides when to invoke ingestion.

[ingest_range](../../memory_v3/agent.py#L150) repeats ingestion across dated sessions. Its normal batch path leaves summary building to a later caller. [build_summaries](../../memory_v3/agent.py#L273) runs the inherited summarizer over all stored dates. Alternatively, ingest can request an incremental summary update for the affected week, month, year, and lifetime. Summary generation makes additional LLM calls. ([Summary update](../../memory_v2/summarizer.py#L228))

The resulting summaries are retrievable through the reader's tools. The current chat invocation supplies the static system prompt and messages; it does not automatically insert the lifetime summary. ([Reader construction](../../memory_v3/agent.py#L118), [invocation](../../memory_v3/agent.py#L287))

## Live conversation: two memory horizons

[converse(message)](../../memory_v3/agent.py#L206) converts every pending turn into a chat message, appends the new user message, and calls the same reader. After the answer, it appends both the user turn and assistant answer to _pending. This lets the reader see something said earlier today before it reaches persistent storage.

[flush(date)](../../memory_v3/agent.py#L236) hands all pending turns to ingest and enables incremental summaries by default. After a successful flush, future calls can retrieve those turns and extracted records from the store. Plain chat does not replay _pending. The buffer exists only in this process and is cleared before ingest returns, so reliable live use depends on calling flush and handling failures. ([Buffer handoff](../../memory_v3/agent.py#L258))

**In one sentence:** the caller controls *when memory is written*; the single reader agent controls *which stored evidence to inspect while answering*.
