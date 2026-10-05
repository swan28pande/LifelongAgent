# Monthly evaluation decisions

Accepted by the user on October 3, 2026.

1. Ingest conversations chronologically, one calendar month at a time. Each user
   has a separate persistent database that carries forward across months.
2. Finish the current month's memory construction and evaluation before ingesting
   any conversations from the next month.
3. At each checkpoint, repeat every question introduced at an earlier checkpoint
   and add the questions introduced this month. Scheduled retrospective copies
   remain distinct question IDs, just as in the dataset.
4. A repeated question retains its **original probe date, wording, reference answer,
   and accepted answers**. "Current" and "now" refer to that original probe date.
   For example, a March question about current residence still asks about March
   when repeated in April. New April questions can ask about April's current state.
5. Use the complete `memory_v4` agent with hierarchical summaries and distillation.
   Reuse the existing `OursV4D` evaluation adapter: its finalization calls
   `build_summaries(force=True, distill=True)`. Agent behavior, models, embeddings,
   retrieval planning, five-tool budget, and judge rules remain unchanged.
6. Evaluation questions, reference answers, generated answers, and judge feedback
   are not ingested into memory. Only the conversation data updates the database.
7. Keep this setup's implementation, documentation, tests, results, and temporary
   files in `setup_2/`. Do not modify the dataset or the existing evaluation setup.

The loader preserves supplied questions and references. Dataset corrections are
the user's separate work; this setup does not rewrite or silently filter them.

The last available partial month is also evaluated. With the 730-day dataset this
includes February 28, 2028. The repaired probe files now introduce new questions
there, alongside repetitions of the accumulated questions. Dataset repairs were
subsequently authorized separately; the loader still preserves supplied records.

## Baseline extension: October 4, 2026

The user requested the three baselines under `setup_2/baselines/` in this same
monthly protocol. All earlier question/date/observation and judging decisions
apply unchanged. `ours_v4d` remains the default, with summaries and distillation.

1. Select `timem`, `naive_rag`, or `full_context` through the common runner.
   DirectPrompting means full context. Results and stores are separate for each
   method, run and user; the method/profile belongs in the run manifest.
2. Persist only observations and method-built memory. Restore incomplete builds
   from the prior checkpoint; reuse complete memory for answer/judge recovery.
3. User processes may run independently in parallel. Within each user, ingestion,
   month finalization, answers and grading remain separated by sequential barriers.
4. NaiveRAG appends raw five-turn chunks to a saved FAISS index. Full context
   includes every observed session and fails before evaluation if its configured
   approximate context budget is exceeded, rather than discarding old sessions.
5. TiMem follows the paper's memory and recall (profile `timem_complexity_aware`):
   2-turn L1 fragments written with the session's three previous fragments, L2-L5
   summaries written with the three most recent earlier summaries of their level, and
   the native complexity-aware recall (native planner; 0.9 dense + 0.1 BM25 L1 ranking;
   bottom-up parents per the strategy's layers and limits; native memory refiner). It
   uses the common selected model at temperature 0.0, durable local node/vector storage,
   cached unchanged summaries, and partial weeks split at calendar month edges. Storage
   is local (SQLite + in-process Qdrant) rather than the native PostgreSQL/Qdrant services,
   and answers use the shared answer prompt like the other baselines.

Original agents, datasets, baseline implementations outside `setup_2/`, and shared
experiment/judge code are unchanged. This extension prepares execution and is
verified offline; it does not initiate a paid benchmark.
