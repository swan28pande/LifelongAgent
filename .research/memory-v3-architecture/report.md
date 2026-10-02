# How memory_v3 works, and what its experiments show

**Status:** Architecture walkthrough complete from current code. The central research claim and whether to design a decisive follow-up experiment remain open choices. This report uses current source for behavior, saved outputs for historical measurements, and primary papers for prior-work comparison. It does not present an illustrative tool route as an observed model trace.

## 1. The mental model

The name AgenticMemoryAgent can suggest a team of agents. The current implementation has **one tool-calling agent**, and it operates on the read side. Writing and summarization are fixed workflows.

~~~text
Batch write
  dated turns
    -> extract typed records with one LLM call
    -> normalize similar values -> remove same-day repeats
    -> SQLite dated memories
    -> five-turn raw chunks in SQLite and FAISS
    -> optional week -> month -> year -> lifetime summaries

Read
  question -> one reader agent -> read-only tools -> evidence -> answer

Live use
  converse() -> today's turns in process memory
  flush(date) -> batch write, then summary update
~~~

[Agent construction](../../memory_v3/agent.py#L100) wires together a [MemoryStore](../../memory_v2/store.py#L35), [IngestionPipeline](../../memory_v3/ingest.py#L115), [Summarizer](../../memory_v2/summarizer.py#L191), and six [read tools](../../memory_v3/tools.py#L225). It makes separate model wrappers for extraction, answering, and summarization, with temperatures 0, 0.3, and 0.2. They can point to the same named model; there is no training or weight update here. The changing state is the stored memory.

The default named model is Gemini. The [model factory](../../memory_v3/agent.py#L53) uses Gemini through Vertex AI unless environment settings select its Developer API, and uses ChatOpenAI for a non-Gemini name. The read agent is a LangChain create_agent graph with a recursion cap of 60 graph steps; that number is not a promise of 60 retrievals.

### Terms that matter

- **Preference:** a specific choice made for a date from a reusable category. “Black coffee on March 8” qualifies; a vague timeless taste statement does not.
- **Fact:** a relatively stable attribute, recorded again when confirmed or changed.
- **Event:** a one-time or scheduled occurrence assigned to its occurrence date.
- **Entity:** the category or topic label attached to a structured record. The extractor chooses it from conversation language; it is not a fixed schema of coffee, clothing, or exercise.
- **Summary:** derived prose at a week, month, year, or lifetime level. It is replaceable when rebuilt.

These definitions are in the [extraction prompt](../../memory_v3/prompts.py#L20). They matter because a row's date is when the choice or event **applies**, which can differ from the day the conversation was ingested.

## 2. Follow one day through the write path

The synthetic dataset gives a concrete transition. On March 7, the user describes an oat milk latte; on March 8, black coffee. A question asks when the first switch happened, with **2026-03-08** as its gold date. ([Conversations](../../datasets/eval/conversations.json), [question](../../datasets/eval/qa_pairs.json))

The [benchmark runner](../../benchmarks/synthetic/run_v3.py#L115) loads sessions, sorts dates, and calls [ingest_range](../../memory_v3/agent.py#L150). That method wraps each date's turns and calls ingest. By default batch ingestion does not update summaries after each day; the runner builds them after all 60 days.

For March 8, [IngestionPipeline.run](../../memory_v3/ingest.py#L122) does the following:

1. **Format.** It creates a transcript with the date, time-of-day labels, and speaker-prefixed turns. The known-block helper reads existing entities and values so the model can reuse spellings.
2. **Extract.** One model call returns JSON arrays of preferences, facts, and events. If the model extracts the coffee choice, one possible item is shown below. This is an *illustrative member* of a response, not the observed full response of the current model:

   ~~~text
   Input transcript day: 2026-03-08
   Possible extracted item:
     entity  = coffee order
     content = black coffee
     date    = 2026-03-08
     type    = preference
     speaker = omitted by the prompt's required JSON schema
   ~~~

3. **Consolidate.** The code compares the value with earlier values under the same entity. It uses token overlap, a subset bonus, and character similarity; a score of at least 0.75 can replace the new text with the existing spelling. This reduces false “changes” caused by wording drift, while risking an overly broad merge. ([Code](../../memory_v3/ingest.py#L218))
4. **Dedupe.** It removes exact same-day entity/value repeats within the extracted batch and against stored rows. A repeat on a *different* day remains, because frequency is evidence for a routine. ([Code](../../memory_v3/ingest.py#L275))
5. **Write.** It inserts surviving structured items into the SQLite memories table. The row stores content, entity, type, speaker, and one date. The default speaker is user when extraction omits it. ([Write](../../memory_v3/ingest.py#L312), [table](../../memory_v2/store.py#L55))
6. **Index raw evidence.** It divides the complete conversation into five-turn chunks, including assistant turns. March 8 has 16 turns, so it produces four chunks. Each goes into the conversations SQL table for exact-date lookup and the conversation FAISS index for semantic search. This step runs even if extraction returns no records or reports an extraction error. ([Chunking](../../memory_v3/ingest.py#L338), [dual storage](../../memory_v2/store.py#L231))

The pipeline returns an [IngestReport](../../memory_v3/ingest.py#L87) with extracted, added, duplicate, consolidated, and indexed counts plus errors. “One LLM call per day” means **one extraction call per pipeline run**. A caller can invoke it more than once in a day; summary building also makes model calls.

### Where each representation lives

| Representation | Persistence | Strongest use |
|---|---|---|
| Typed dated records | SQLite memories table | Exact date, ordering, counting, changes |
| Raw five-turn chunks | SQLite conversations table and FAISS | Exact-day wording or semantic topic search |
| Period summaries | FAISS summaries index; hashes in SQLite | Broad profile and long-range patterns |

The store uses [nomic-ai/nomic-embed-text-v1](../../memory_v2/store.py#L20) with document/query task prefixes for FAISS. Structured memory rows are queried with SQL rather than embedded. There is no direct provenance key from a structured row to its supporting raw chunk; the reader can recover nearby conversation by date.

## 3. Summaries are a separate derived layer

The inherited [Summarizer](../../memory_v2/summarizer.py#L206) constructs summaries per speaker in this order: week, month, year, lifetime. It asks a model to identify recurring preference entities, analyzes transitions per entity, and combines facts and events. Higher levels consume lower-level summaries. It saves the result in the summary FAISS index.

There are two cadences:

- **Full build:** build_summaries traverses the date range and builds every level. The synthetic and LoCoMo benchmark runners call this after ingesting their conversations.
- **Incremental update:** ingest with update_summaries enabled follows the affected week → month → year → lifetime chain. Source hashes skip unchanged levels. Live flush enables this by default. ([Agent](../../memory_v3/agent.py#L129), [dirty checks](../../memory_v2/summarizer.py#L228))

An important code/document difference: the lifetime-summary generation prompt says that profile will be injected into future chats, but [chat construction](../../memory_v3/agent.py#L118) supplies only the static reader prompt and retrieval tools. A summary can be found through a tool; it is **not automatically inserted**. Weekly summarization fetches raw conversations but does not pass their text to its generation prompt; the weekly analysis uses structured records. ([Weekly code](../../memory_v2/summarizer.py#L308))

## 4. Follow the question through the read path

[chat](../../memory_v3/agent.py#L180) passes one HumanMessage into the read agent. Its [system prompt](../../memory_v3/prompts.py#L95) suggests summaries for broad profile questions, structured memories for chronology and counts, raw conversation search for discussion detail, and exact-date lookup when a date is named. The loop is the standard LangChain model/tool loop: the model receives messages and tool descriptions, may emit one or more tool calls, receives their returned text as new messages, and repeats until it emits a final answer or reaches the recursion cap. The application supplies no separate custom planner or controller.

| Tool | Source | Typical reason to call |
|---|---|---|
| list_entities | SQLite distinct labels/speakers | Discover actual storage names |
| search_memories | SQLite dated records | Reconstruct an exact timeline |
| semantic_search_conversations | Conversation FAISS | Find discussion by meaning |
| read_conversations_on | Conversation SQL | Inspect an exact day, optionally ±7 days |
| semantic_search_summaries | Summary FAISS | Find broad context by meaning |
| get_summary | Summary FAISS by identifier | Fetch one known period |

For “When did the coffee choice first switch?”, a **plausible** current-code route is list_entities, then search_memories for that entity across early March, optionally followed by read_conversations_on for March 7 and 8. The model could choose another route. [chat_with_trace](../../memory_v3/agent.py#L184) exposes tool names and arguments, but not tool outputs, so its name “full tool-call trace” overstates the evidence it returns.

The tool surface is read-only. The reader cannot add, remove, or correct memories during chat. This matters when reading the older [PROJECT_PLAN](../../PROJECT_PLAN.md): it proposes writer tools and an agentic write process, while the implemented v3 writer is a fixed pipeline.

The tools are built as closures over the store. The model sees their names, descriptions, arguments, and returned strings; it does not receive a MemoryStore object or direct SQL access. This interface boundary is why tool descriptions matter to retrieval behavior. ([Tool factory](../../memory_v3/tools.py#L39))

## 5. Live use differs from the benchmark

[converse](../../memory_v3/agent.py#L206) includes all pending turns from today in the next reader invocation, then appends the new user message and answer to an in-process list. Thus a breakfast preference can affect a dinner answer even before storage changes. [flush](../../memory_v3/agent.py#L236) clears that list, ingests the day's turns, and updates summaries unless disabled. A plain chat call includes no pending turns. The inspector [API chat endpoint](../../app/backend.py#L259) calls chat rather than converse.

Two operational limits follow directly from code. Pending turns exist only in process memory, and flush clears the buffer before ingest returns; a failed flush can lose that buffer. Reingesting a date deduplicates structured records but adds its raw conversation chunks again. These are implementation risks, not measured rates of failure. ([Agent](../../memory_v3/agent.py#L258), [index](../../memory_v3/ingest.py#L338))

The prompt's JSON schema does not require a speaker field; the pipeline otherwise assigns the ingest-call speaker. Turn speaker labels therefore do not guarantee per-person structured attribution in two-person dialogue. Also, a future event's effective date can differ from the transcript date, yet incremental summary updates follow the ingest date. ([Prompt](../../memory_v3/prompts.py#L85), [write](../../memory_v3/ingest.py#L318), [update](../../memory_v3/agent.py#L145))

## 6. What the experiments establish

### Synthetic preference evolution

The [generator](../../scripts/generate_eval_dataset.py#L47) defines three daily schedules for one synthetic user: coffee changes after blocks of seven days, clothing after blocks of three days, and exercise follows weekdays and weekend states. It generates 60 daily conversations and 67 questions. The question mix is 22 labeled factual, 24 recall, 9 transition, 9 future prediction, and **3 direct pattern-identification** items. A full two-state coffee repeat takes 14 days even though one description calls it a “7-day cycle”; the wording should be read as a seven-day *switch interval*. ([Generator](../../scripts/generate_eval_dataset.py#L348))

The archived v3 run reports **0.143 token F1**, **0.478 LLM-judge accuracy**, and **0/3** direct pattern questions correct. Its judge accepts a core correct item inside extra or verbose text, so judged prediction success alone does not prove the rule was inferred correctly. This is one user and three generated schedules, with questions asked after all 60 days have been ingested. It tests retrospective QA and some future-choice answers, not live online adaptation. ([Results](../../results/synthetic/v3/README.md), [judge prompt](../../benchmarks/synthetic/run_v3.py#L39))

The saved result is from an older revision: it names gemini-3.1-flash-lite, whereas the current runner defaults gemini-3.5-flash; saved rows lack the current type field. The legacy comparison ingested 30 days but answered the full 67-question set, while v3 ingested 60; the token-F1 implementations also differ. A direct v3-versus-legacy ranking would mix evidence windows and metrics. ([Archived output](../../results/synthetic/v3/qa_results.json), [runner](../../benchmarks/synthetic/run_v3.py#L102), [legacy evaluator](../../evaluation/evaluate_synthetic.py#L575))

### LoCoMo transfer check

The saved [LoCoMo comparison](../../results/locomo/README.md) covers one conversation, conv-26, with 199 questions. It reports overall paper-labeled F1 of 0.460 for v3 agentic, 0.608 for v3 one-pass, and 0.557 for local mem0; excluding adversarial questions, the figures are 0.557, 0.546, and 0.452. The agentic reader uses roughly eight answer-time model calls per question; the other two use one.

These scores do **not** isolate the benefit of agentic retrieval. Agentic and one-pass v3 share a store, but use different answer prompts and budgets. V3 one-pass and mem0 share a prompt, but their stores and context amounts differ substantially. The 2×2 design lacks a mem0-store/agentic-reader cell. ([Protocol](../../benchmarks/locomo/README.md#L19), [runners](../../benchmarks/locomo/run_v3_onepass.py#L40))

The local scorer is also not paper-exact as labeled: it accepts “don't know” for an adversarial question, while the bundled LoCoMo evaluator accepts only “no information available” or “not mentioned”; that evaluator also truncates category-3 answers at a semicolon. The saved agentic trace refers to search_preferences, while the current code exposes search_memories. Those artifacts are useful history, not a pinned evaluation of today's code. ([Local scoring](../../benchmarks/locomo/common.py#L62), [bundled evaluator](../../baselines/locomo/task_eval/evaluation.py#L203), [current tool](../../memory_v3/tools.py#L68))

The [offline tests](../../tests/test_memory_v3.py#L1) support the plumbing claim: fixed extraction responses flow into SQL and FAISS, tool outputs have the expected shape, and the live buffer behaves as coded. Fake embeddings and scripted model responses do not test extraction quality, retrieval quality, or pattern reasoning.

## 7. Prior work, compared on the same dimensions

- **[APEX-MEM](https://aclanthology.org/2026.acl-long.749.pdf):** Addresses evolving conversational facts with append-only temporal graph records, source evidence, and multi-tool retrieval including structured queries. It overlaps strongly with v3's broad structured-history-plus-reader idea, while using a richer graph, validity intervals, and provenance rather than v3's flat dated choice rows.
- **[Chronos](https://arxiv.org/html/2603.16862):** Keeps extracted temporal events and raw turns in separate calendars, then lets an agent search both. It overlaps with v3's raw-plus-structured temporal retrieval, while focusing on event tuples and query-specific guidance rather than v3's preference/fact/event ledger and calendar summaries.
- **[TiMem](https://aclanthology.org/2026.findings-acl.1091.pdf):** Consolidates conversation into segment, session, day, week, and profile memories, with recall guided by query complexity. It overlaps with temporal hierarchy and behavioral-pattern summaries, while lacking v3's documented exact per-day choice ledger and SQL timeline interface.

These are architectural comparisons from the primary papers. Their datasets, models, answer budgets, and scorers differ from the local runs, so their reported percentages do not rank these systems against this repository. The broader [prior-art report](../memory-v3-prior-art/report.md) covers additional systems.

## 8. Current assessment and open decisions

The implementation clearly provides three complementary evidence levels: exact dated rows, original conversation chunks, and broader period summaries. The current tests verify that those components connect. The archived experiments provide weak evidence for the stronger claim that the hierarchy discovers periodic choices: there are only three direct pattern questions, and the saved run misses all three. The LoCoMo comparison provides a useful second setting but does not isolate reader strategy or establish generalization from one conversation. These are interpretations of the cited evidence, not measured properties of the exact current revision.

Two decisions remain for this walkthrough:

1. Which claim should be the main target: faithful changing-choice histories and cycle reasoning, advantage of a tool-calling reader, or broad lifelong assistant quality? The first is the most precise fit to the project's stated problem.
2. Should the assessment end with a matched, falsifiable experiment design? That would specify held-out users and schedules, equal context and model budgets, a common scorer, and separate measures for extraction, exact history, cycle inference, prediction, abstention, cost, and live updates.

The detailed source-by-source observations are in [evidence.md](evidence.md); the investigation scope is in [plan.md](plan.md).
