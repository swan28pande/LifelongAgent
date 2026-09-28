# Method breakdowns: industry and production memory systems

Reference: the researcher's method.
1) LLM extraction of preferences, facts and events from each day's conversation (dated records)
2) SQL database of records (entity, value, type, date)
3) RAG (vector search) over raw conversation chunks and summaries
4) Hierarchical summarization (weekly, monthly, yearly, lifetime)
5) Agentic tool-calling retriever that decides which store to query

All statements are from the papers in `papers/`. "Not specified in paper" means the paper does not say.

---

## Mem0 and Mem0^g (ECAI, 2025; arXiv 2504.19413)

**Fundamental components**
1. **Incremental LLM extraction per message pair.** For each new pair (m_{t-1}, m_t), an LLM gets three inputs: a conversation summary S, the last m=10 messages and the new pair. It outputs a set of candidate "salient memories", which are natural-language facts. The summary S is refreshed asynchronously by a separate module.
2. **Vector store of natural-language facts.** Dense embeddings go in a vector database. The final answer prompt calls the memories "timestamped", but the schema of those timestamps is not specified in the paper.
3. **LLM tool-call update.** For each candidate fact, the system retrieves the top s=10 similar memories. The LLM then picks ADD (new fact), UPDATE (replace with richer info), DELETE (remove the contradicted memory) or NOOP. Deletion is physical removal (Algorithm 1).
4. **Mem0^g graph variant (Neo4j).** An entity extractor and a relation generator produce (source, relation, destination) triplets. Each node has a type, an embedding and a creation timestamp. A conflict detector plus an LLM update resolver mark conflicting edges as **invalid** instead of deleting them.
5. **Retrieval.** Mem0^g uses two paths: entity-centric retrieval (anchor nodes, then a subgraph of incoming and outgoing edges) and dense matching of the whole query against triplet text with a threshold. Query-time retrieval for base Mem0 is not detailed beyond memory search. No hierarchical summarization.

**Handles changing facts/preferences?** Base Mem0 overwrites (UPDATE) or hard-deletes (DELETE). Mem0^g marks old edges invalid. The answer prompt says to prefer the most recent memory.

**Overlap with the researcher's method:** Shares 1 (LLM fact extraction) and 3 (vector search, but over extracted facts, not raw chunks). Partly shares 5 (tool calls are used for writing memory, not for choosing a store at retrieval). Unlike the researcher's append-only dated SQL records, Mem0 edits one mutable fact store in place. It has no hierarchical summaries and no SQL.

---

## Zep / Graphiti (arXiv, 2025; 2501.13956)

**Fundamental components**
1. **Episodes (raw, non-lossy).** Each raw message is stored with a reference timestamp t_ref as an episode node. Episodic edges link it to the entities it mentions, so every fact can be traced back to its source.
2. **LLM entity and fact extraction into a knowledge graph (Neo4j).** Context is the current message plus the last n=4 messages, with a reflection step. Entities are resolved against existing nodes using embedding cosine search plus full-text search, then an LLM resolution prompt. Facts become edges, deduplicated among edges between the same entity pair. Writes use predefined Cypher queries.
3. **Bi-temporal model.** Each edge has t'_created / t'_expired (ingestion timeline T') and t_valid / t_invalid (event timeline T). Relative dates are resolved using t_ref.
4. **Edge invalidation.** An LLM compares a new edge with semantically related edges. On a temporally overlapping contradiction it sets the old edge's t_invalid to the new edge's t_valid. History is kept, and new information wins.
5. **Community subgraph.** Label propagation clusters entities, with a dynamic single-step extension and periodic refreshes. Communities get map-reduce summaries and embedded names.
6. **Retrieval pipeline f = constructor(rerank(search(q))).** Search combines cosine similarity, BM25 (Lucene) and n-hop BFS over facts, entity names and community names. Reranking uses RRF, MMR, episode-mentions frequency, node distance or a cross-encoder. The output is a context string listing facts with their valid date ranges, plus entity summaries (top 20 in experiments).

**Handles changing facts/preferences?** Yes, by invalidating old edges with a validity window (t_valid/t_invalid). Nothing is deleted.

**Overlap with the researcher's method:** Shares 1 (dated fact extraction), 2 in spirit (structured, dated records, but in a graph DB rather than SQL) and 3 (hybrid vector plus BM25). Community summaries are a topical hierarchy, not a temporal week/month/year one, and retrieval is a fixed pipeline, not an agent (no 5). Its explicit validity intervals go beyond the researcher's single `date` field.

---

## MemGPT / Letta (arXiv, 2023; 2310.08560)

**Fundamental components**
1. **OS-style memory hierarchy.** Main context (prompt tokens) is split into read-only system instructions, a **working context** and a **FIFO message queue**. External context has two stores: **recall storage** (a database of all messages) and **archival storage** (a read/write database of arbitrary-length text).
2. **Self-directed writes via function calls.** The LLM decides what to store. It appends or replaces key facts, preferences and persona info in working context (e.g. `working_context.replace("Boyfriend named James", "Ex-boyfriend named James")`) or writes to archival storage. No separate extraction pipeline is used.
3. **Queue manager and recursive summary.** At about 70% of the context window, a "memory pressure" warning prompts the LLM to save important info. At the flush limit, about 50% of messages are evicted and a new **recursive summary** is built from the old summary plus the evicted messages. Evicted messages stay in recall storage indefinitely.
4. **Agentic retrieval.** The LLM calls functions such as `recall_storage.search(...)` and `archival_storage.search(...)` with paginated results. Function chaining (`request_heartbeat=true`) allows multi-step and multi-hop lookups. Archival storage uses PostgreSQL with pgvector (cosine similarity, HNSW index). The search method for recall storage is not specified in paper.
5. **Event-driven control flow.** User messages, system alerts and timed events trigger inference.

**Handles changing facts/preferences?** The LLM overwrites working-context text itself (replace). No validity windows or versioning are specified in paper.

**Overlap with the researcher's method:** Shares 5 most strongly (an agent calling tools to pick a store) and 3 (vector search over archival text). It has one rolling recursive summary, not time-bucketed hierarchical summaries. Memory is free text, not structured (entity, value, type, date) rows, so it lacks 1 and 2 as systematic components.

---

## EverMemOS (arXiv, 2026; 2601.02163)

**Fundamental components**
1. **Episodic Trace Formation.** An LLM semantic boundary detector segments the dialogue stream over a sliding window by topic shift. Each segment is rewritten into a third-person **Episode** narrative with coreferences resolved.
2. **MemCell = (Episode, Atomic Facts, Foresight, Metadata).** Atomic facts are discrete, verifiable statements. Foresight holds forward-looking inferences (plans, temporary states) with validity intervals [t_start, t_end]. Metadata holds timestamps and source pointers. The storage backend is not specified in paper.
3. **Semantic Consolidation into MemScenes.** Online incremental clustering assigns each new MemCell to the nearest MemScene centroid if similarity > tau (subject to a max time-gap in days), or else creates a new scene. The scene summary is updated on assimilation.
4. **User Profile.** It has two parts: explicit facts (including time-varying measurements) and implicit traits. It is refreshed from scene summaries with "recency-aware updates" and "conflict tracking" (Appendix B.3). Deletion or forgetting of MemCells is not specified in paper.
5. **Reconstructive Recollection (agentic retrieval).** Dense (Qwen3-Embedding-4B) and BM25 retrieval over atomic facts are fused with RRF, and scenes are scored by their max MemCell score (top N=10). Episodes from those scenes are pooled and reranked (Qwen3-Reranker-4B, K=10). Expired Foresight is filtered by t_now. An LLM sufficiency check can trigger query rewriting for another retrieval round (31% of LoCoMo questions).

**Handles changing facts/preferences?** Partly. The profile uses recency-aware updates with conflict tracking, and Foresight expires through validity intervals. MemCells themselves appear to be append-only (no update rule is stated).

**Overlap with the researcher's method:** Shares 1 (LLM extraction of dated facts and events), 3 (hybrid vector plus BM25 retrieval) and 5 in part (an LLM sufficiency and rewrite loop, but not tool calls choosing among stores). Consolidation is topical (MemScenes plus profile), not temporal week/month/year summaries, and there is no SQL store.

---

## SimpleMem (arXiv, 2026; 2601.02553)

**Fundamental components**
1. **Semantic Structured Compression.** Dialogue is cut into overlapping sliding windows (W=20). One LLM pass does implicit density gating (an empty output discards filler) and "de-linearization" into self-contained **memory units**: coreference resolved, relative time converted to absolute ISO-8601 timestamps, and complex turns split into atomic facts. Each unit has content, entities, topic, timestamp and salience.
2. **Multi-view indexing.** Each unit is indexed three ways: a dense embedding (Qwen3-embedding-0.6b), a BM25 sparse/lexical index and symbolic metadata (timestamps, entity types) in SQL-based storage. The implementation uses LanceDB.
3. **Online Semantic Synthesis.** Within a session, at write time, the LLM merges related fragments into one consolidated entry (e.g. three coffee facts become "User prefers hot coffee with oat milk") before they are committed. These act as "abstract representations" alongside the detailed units.
4. **Intent-Aware Retrieval Planning.** From the query and history, the LLM produces a semantic query, lexical keywords, symbolic/temporal constraints and a retrieval depth d (k from 3 to 20). The three views are queried in parallel, and the results are combined by set union with ID deduplication.
5. Cross-session consolidation, decay or deletion: not specified in paper.

**Handles changing facts/preferences?** No explicit update or invalidation at write time. Units are appended, and at answer time the prompt says "prioritize the most recent memory unit".

**Overlap with the researcher's method:** Shares 1 (LLM extraction of timestamped facts), 2 (SQL metadata store with entity and timestamp fields) and 3 (vector search, plus BM25). Retrieval is planned in a single LLM call rather than by an iterative tool-calling agent (a weak 5), and it has no temporal hierarchical summaries (4). Its closest analogue is intra-session merge synthesis.
