# Method breakdowns: temporal / hierarchical memory competitors

Reference, the researcher's method (R1-R5):
R1 LLM extraction of preferences, facts and events from each day's conversation (dated records);
R2 SQL database of records (entity, value, type, date);
R3 RAG (vector search) over raw conversation chunks and summaries;
R4 hierarchical summarization, calendar-aligned (weekly -> monthly -> yearly -> lifetime);
R5 agentic tool-calling retriever that decides which store to query.

All statements below come from the papers in `papers/`. "Not specified in paper" means I could not find it in the text.

---

## TiMem (ACL Findings, 2026)

**Fundamental components**
1. **Temporal Memory Tree (TMT)** with 5 levels: L1 segment, L2 session, L3 day, L4 week, L5 profile. Every node stores a time interval tau(m) = [t_start, t_end] and a semantic memory sigma(m), kept as text plus embeddings. A parent's interval must contain its children's intervals ("temporal containment").
2. **The levels are calendar/session-aligned.** L1 is created online after every user-assistant turn (w_d = 1). L2-L4 are consolidated when a session, day or week window closes. The L5 profile is updated monthly. The time used is the time of the dialogue. The paper does not separate the time an event happened from the time it was mentioned.
3. **Level-specific consolidation prompts.** The consolidator prompt Phi_i takes three inputs: the child memories, the 3 most recent memories at the same level, and a level instruction. L1-L2 do factual summarization, L3-L4 do "evolving patterns" (routines, behaviour and preference patterns), and L5 is the persona (personality, preferences, values). No fine-tuning.
4. **Recall planner.** An LLM labels the query as simple, hybrid or complex and extracts keywords. The label decides which levels are searched: simple searches fact + profile; hybrid searches fact + part of the pattern levels + profile; complex searches all levels.
5. **Hierarchical recall.** L1 leaves are scored with lambda * cosine + (1 - lambda) * BM25 (lambda = 0.9). The system then adds the leaves' ancestors at the planned levels. An LLM "recall gating" step drops irrelevant or conflicting memories. The rest are sorted by level and by distance from the query time.

**Handles changing facts/preferences?** Only through consolidation. The weekly and profile levels summarize "evolving" preferences, and the gating step filters conflicts at recall time. There are no explicit update or invalidation operations.

**Overlap:** R3 (partial) and R4. TiMem also has a calendar hierarchy (day -> week -> monthly profile), but it summarizes raw dialogue segments. It has no structured, dated fact/preference records (R1/R2), no SQL store, and no yearly or lifetime levels. Its planner picks levels from a fixed complexity label, not with an agent choosing tools (R5).

---

## MemForest (arXiv, 2026)

**Fundamental components**
1. **Parallel chunk extraction.** Each session is split into fixed-size chunks, and an LLM extracts facts from the chunks in parallel. A canonicalization step then merges equivalent facts by exact match, embedding candidates and a bounded LLM equivalence check. Each canonical fact holds fact text, "temporal anchors", entity labels and scene labels, and is stored in a Fact Manager.
2. **Routing into scopes.** Each fact goes to three kinds of scope: session (its source dialogue), entity (normalized entity labels) and scene (cosine clustering of facts against cluster centroids). A fact can belong to more than one scope.
3. **MemTree (one per scope).** This is a balanced k-ary temporal tree. Leaves are time-ordered facts (entity/scene trees) or dialogue units (session trees). Internal nodes summarize contiguous time intervals, and the root summarizes the whole tree. **The levels come from tree depth under bounded fan-out, not from calendar units.** The paper does not say whether a temporal anchor is the event time or the time of the conversation.
4. **Maintenance by eager structure, lazy summaries.** New facts are appended as leaves and the tree is rebalanced (O(log_k N)). The ancestors of changed leaves are marked dirty and re-summarized bottom-up in batches. The same mechanism handles merge, delete and migration.
5. **Retrieval.** The system first picks candidate trees ("forest recall"): the top-k root-summary embeddings plus the trees that contain the top-k atomic facts. It then browses each tree from the root down to the leaves, choosing children either by embedding similarity alone or with an LLM. Optionally, a planner writes a sub-query for each tree.

**Handles changing facts/preferences?** Superseded states are kept as their own time-anchored leaves. Contradictory updates are not merged and stay as separate facts, so the history remains queryable.

**Overlap:** R1 (partial), R3 (partial) and R4 (partial). MemForest also extracts dated facts and summarizes them hierarchically. Its hierarchy is a balanced tree per session, entity or scene, with levels set by fan-out and depth, not by week, month or year. It has no SQL table (R2), and its retrieval is fixed tree traversal, not an agent choosing stores (R5).

---

## TReMu (ACL Findings, 2025)

**Fundamental components**
1. **Time-aware memorization (timeline summarization).** An LLM writes a summary of each session, stamped with the session date. It also writes a separate short summary for every event mentioned in the session, dated with the inferred event time. For example, "last week" said on 08/03/2022 becomes "During 07/24/2022 to 07/30/2022". The paper explicitly separates when an event happened from when it was mentioned.
2. **Memory organization as a timeline.** Memory pieces are grouped by events that happen at the same time and indexed by the inferred timestamp. The storage backend is not specified in paper. The pipeline is built on MemoChat.
3. **Retrieval.** An LLM reads the query and the numbered memory options and selects the relevant ones. The paper describes no vector search.
4. **Neuro-symbolic temporal reasoning.** An LLM writes Python (datetime, dateutil, plus helpers such as weekRange and lastWeekendRange) from the retrieved memory. The code is run, and its output is given back to the LLM to choose the answer.
5. **No hierarchy.** There is one timeline level. The paper describes no consolidation or update step.

**Handles changing facts/preferences?** Not addressed. Changes can only show up as separate dated events on the timeline.

**Overlap:** R1 (partial). TReMu also creates dated event records and separates event time from session time. However, it only summarizes events: it does not extract preferences or entity-value facts, and it has no SQL store (R2), no vector RAG (R3) and no hierarchy (R4). Its retrieval is a single LLM selection step followed by code generation, not a multi-store tool agent (R5).

---

## TSM, Temporal Semantic Memory (ACL Findings, 2026)

**Fundamental components**
1. **Episodic memory as a Temporal Knowledge Graph (TKG), built following Zep.** Entities and relations are extracted from each turn together with the preceding n turns. They are stored as facts (e_s, r, e_o, t), where t is the **semantic (event) time**, not the dialogue time. Each fact has valid_time and invalid_time. Each entity node keeps an LLM-written entity summary. A bidirectional index links entities to their source chat turns.
2. **Online graph update.** New entities are either added or merged into an existing entity. New facts are compared with existing edges by meaning and by time, and one of four operations is applied: DUPLICATE, ADD, INVALIDATE or UPDATE.
3. **Durative memory.** The TKG is cut into **fixed calendar slices of one month by default**. Within each slice, a GMM clusters entities. For each cluster the LLM writes a Topic (a summary of the entity summaries) and a Persona (from the linked dialogue: stable traits, preferences, behaviour patterns). Each entry carries its slice timestamp and an embedding. There are only two levels: episodic facts and monthly topic/persona.
4. **Sleep-time consolidation.** Topic and persona summaries are rebuilt periodically, for example monthly or when enough new turns have accumulated, by re-clustering and re-summarizing.
5. **Retrieval.** spaCy parses the query's time constraint T_q. Dense retrieval runs over topics, personas and raw chat turns. Topics and personas outside T_q are removed. Chat turns linked to TKG facts valid within T_q are added. Results are ranked first by whether they fall inside T_q, then by similarity.

**Handles changing facts/preferences?** Graph edges carry validity intervals, and the INVALIDATE and UPDATE operations mark facts that no longer hold. Personas are rebuilt for each month, so they can change from month to month.

**Overlap:** R1 (partial), R3 and R4 (partial). TSM also uses calendar-aligned summaries (monthly), personas that carry preferences, and vector RAG over raw turns and summaries. Its store is a Zep-style temporal knowledge graph, not an SQL table (R2), and it has only one level above the facts, with no week -> month -> year -> lifetime chain. Its retrieval is a fixed pipeline (parse time, then dense search, then filter), not an agent choosing tools (R5).

---

## APEX-MEM (ACL, 2026)

**Fundamental components**
1. **Ontology-guided fact/event extraction.** For each turn, a few-shot, schema-constrained LLM extracts an event (type, timestamp T, location, participants, facts, evidence). Each fact is (subject, property, value, datatype, [t_from, t_to], confidence, evidence). Times are converted to ISO 8601 relative to the turn's timestamp, so events are dated. The ontology has 35 entity classes.
2. **Entity and property resolution.** Dense top-k candidates are retrieved, and an LLM decides whether a mention matches an existing entity, is a new entity, or neither. Property names are converted to snake_case and given types.
3. **Append-only property graph stored in SQLite.** The tables are entities, properties, facts, events, event_participants, evidence and turns, with a hybrid dense+lexical index. Nothing is overwritten: all versions are kept.
4. **No hierarchical summarization.** An entity document shows its "latest" values and "anchors" as markdown tables, created at query time. An "online" mode builds the graph only from documents relevant to the query.
5. **ReAct agent retriever with tools.** SCHEMAVIEWER shows the schema and usage guidance. ENTITYLOOKUP resolves a surface form to a canonical entity and returns a time-aware fact snapshot. GRAPHSQL runs read-only SELECT queries for joins, ordering and durations. SEARCH does hybrid semantic+lexical retrieval over entities, properties, events and turns. The agent may make at most 40 calls.

**Handles changing facts/preferences?** Resolved at retrieval time. Superseded facts are kept with validity intervals, and the agent picks the most recent valid entry or reasons over the full history using SQL.

**Overlap:** R1, R2, R3 (partial) and R5. This is the closest match on the fact store and the retriever: dated extraction into SQL tables plus a tool-calling agent that uses SQL and hybrid search. APEX-MEM has **no hierarchical or calendar summaries (no R4)**. Its extraction is per turn and ontology-based rather than per day, and it does not treat preferences as a separate record type (not specified in paper).

---

## THEANINE (NAACL, 2025)

**Fundamental components**
1. **Session-level memory summarization.** At the end of each session, an LLM (gpt-3.5-turbo) summarizes it into memories m = (event, time), where time is when the memory was formed, i.e. the session. "Event" covers things done or said and speaker personas. The paper does not separate event time from dialogue time.
2. **Relation-aware memory graph.** Each new memory is compared with its top-j (j = 3) most text-similar memories. An LLM assigns a commonsense relation to each pair (Cause, Reason, Want, HinderedBy, SameTopic, ...) or "None". The new memory is then linked to the most recent related memory in each connected component. Edges also record temporal order.
3. **No memory update or deletion.** Old memories are never removed, on purpose.
4. **Timeline retrieval.** The top-k (k = 3) memories are retrieved by embedding similarity to the current dialogue. The system takes each one's connected component and splits it into linear timelines, running from the oldest memory to an end point. It samples n = 1 timeline per retrieved memory.
5. **Context-aware timeline refinement.** An LLM rewrites the retrieved timelines to fit the current dialogue, and the rewritten timelines are used to generate the response.

**Handles changing facts/preferences?** Changes are kept as linked chains of events over time, never overwritten. The model reads the whole evolution at response time.

**Overlap:** R1 (weak) and R3 (partial). THEANINE also keeps full histories and uses embedding retrieval. It stores memories in a causal/temporal graph, not SQL records, and has no hierarchical or calendar summaries (R4) and no tool agent (R5). Its "timeline" is a graph path, not a date-indexed hierarchy.

---

## Cross-system summary for the novelty claim

| System | Calendar-aligned hierarchy | Structured dated records | SQL store | Vector RAG over raw + summaries | Agent tool retrieval |
|---|---|---|---|---|---|
| TiMem | Yes (session/day/week, monthly profile) | No (summaries) | No | Partly (BM25+dense on L1, then ancestors) | No (complexity planner) |
| MemForest | No (tree depth/fan-out) | Yes (canonical facts) | No | Yes (roots + facts) | No (tree browse, optional per-tree planner) |
| TReMu | No | Events with inferred dates | No | No (LLM selection) | No (code execution) |
| TSM | Monthly slices only (1 level) | Yes (TKG facts with validity) | No | Yes | No |
| APEX-MEM | No | Yes (facts/events with validity) | Yes (SQLite) | Yes (hybrid) | Yes (ReAct, 4 tools) |
| THEANINE | No | No (session event summaries) | No | Partly (top-k seed) | No |

No paper here combines all three of: dated preference/fact/event records in SQL (R1+R2), a multi-level calendar hierarchy going past monthly to yearly and lifetime (R4), and an agent that chooses between SQL, vector and summary stores (R5). The closest pairs are APEX-MEM (R1+R2+R5, but no R4) and TiMem/TSM (calendar R4 up to week or month, but no SQL records and no agent).
