# Prior art and possible successors to `memory_v3/`

Research date: 2026-09-26. Scope: public work available by this date.

## Objective

Identify the papers and systems closest to the current `memory_v3/` implementation, and determine which, if any, substantially subsume its capabilities. The result should help position a research claim or choose stronger baselines and experiments.

## Local system and terms

The unit of comparison is the implementation in `memory_v3/`, including its imported `memory_v2/store.py` and `memory_v2/summarizer.py`, rather than the older claims in `paper.tex`.

- Write: one extraction call per ingested day produces dated, entity-keyed preferences (committed choices), stable facts, and events; deterministic value consolidation and same-day deduplication; append-only rows; five-turn raw transcript chunks.
- Storage: SQLite for dated records and exact conversation date lookup; FAISS for raw conversation and summary search.
- Derived memory: incremental weekly, monthly, yearly, and lifetime summaries, including analysis of recurring preference sequences.
- Read: a tool-calling agent chooses among structured timeline queries, exact date lookup, semantic search of raw turns and summaries, and direct summary lookup.
- Live behavior: current-day turns stay in context, then are ingested on `flush()`.

"Closest" means the greatest overlap in problem, memory semantics, update rules, retrieval, and evaluation target. "Superseding" means a work demonstrably covers the key capabilities and has stronger evidence or a more capable method on the same tasks. A larger score on a different benchmark, model, judge, context budget, or ingestion regime alone does not establish supersession.

The repository has no root `CONTEXT.md` or populated `docs/adr/` at research start. Domain terms above come from the implementation and its README.

## Subquestions and evidence required

1. Which primary works already combine dated, append-only user facts or events with evolving-state resolution and agentic structured retrieval? Inspect method sections and official code for temporal semantics, deletion/overwrite policy, tools, and evaluations. Search under alternate terms such as temporal knowledge graphs, event sourcing, bitemporal memory, user profile evolution, and conversational memory.
2. Which systems combine episodic/raw evidence with hierarchical summaries, semantic memory, and adaptive retrieval? Check whether hierarchy is by calendar period, topic, or event, and whether original evidence remains recoverable.
3. Which practical memory systems (especially Mem0, Graphiti/Zep, Hindsight, LangMem) provide the same or stronger end-to-end capability? Separate papers, currently documented product features, and marketing claims.
4. How comparable are reported results? Record dataset version/size, question categories, model, retrieval budget, judge/metric, and whether ingestion is online or batch. Compare to local LoCoMo conv-26 (199 questions) and 60-day synthetic run only where justified.
5. What important counterexamples or newer work could invalidate a novelty claim focused on temporal preference tracking, append-only histories, hierarchical summaries, or a tool-calling reader?

## Comparison dimensions

Problem; memory types/granularity; temporal model and update semantics; raw-evidence retention; consolidation and hierarchy; retrieval control; model/training requirements; data and evaluation protocol; demonstrated strengths and limitations. Distinguish source facts, authors' claims, and this investigation's inferences.

## Source strategy

Prefer peer-reviewed papers or arXiv originals, ACL Anthology, official repositories, official benchmark papers and data, and official product documentation for current systems. Record exact section, table, or repository location. Use secondary material only to discover primary sources.

## Falsification checks

- Find an earlier or contemporary system with the same append-only preference timeline and SQL/tool-based temporal queries.
- Find a system that does better on recurring preference cycles and evolving facts, not just isolated fact lookup.
- Check if attractive results omit adversarial questions or use a permissive LLM judge.
- Check if local gains disappear with equal answer model, prompt, retrieved context budget, and all LoCoMo conversations.
- Check whether summary-level pattern claims are actually supported by local results.

## Stopping conditions

Stop after primary sources cover the closest temporal/structured, hierarchical/episodic, and practical reference systems; targeted alternate-terminology searches reveal no closer method; and the main conclusion can be stated with specific overlap and uncertainty. Avoid claiming an exhaustive survey or direct performance ranking without a common protocol.
