# Method breakdowns: academic memory systems

Reference (researcher's method): 1) LLM extraction of preferences/facts/events per day (dated records); 2) SQL DB of records (entity, value, type, date); 3) RAG (vector search) over raw chunks + summaries; 4) hierarchical summarization (weekly -> monthly -> yearly -> lifetime); 5) agentic tool-calling retriever that picks which store to query.

Source: PDFs in `papers/`, method sections plus appendices where noted.

---

## A-MEM (NeurIPS, 2025) - `2025_neurips_a-mem.pdf`

**Fundamental components**
1. **Note construction**: each interaction becomes one atomic "memory note" m_i = {c_i original content, t_i timestamp, K_i keywords, G_i tags, X_i contextual description, e_i embedding, L_i links}. The LLM generates K, G and X from the content and timestamp (Sec. 3.1).
2. **Dense embedding store**: e_i = encoder(concat(c, K, G, X)), using all-MiniLM-L6-v2. The paper does not name a database backend.
3. **Link generation**: for a new note, take the top-k notes by cosine similarity, then the LLM decides which of them to link (L_i). This makes a Zettelkasten-style network of "boxes" (Sec. 3.2).
4. **Memory evolution**: for each neighbour note, the LLM may rewrite its context, keywords and tags given the new note, and the evolved note replaces the old one (Sec. 3.3). The prompt's action list includes "strengthen/merge/prune", but the method text never describes deletion or forgetting.
5. **Retrieval**: the query is embedded, the top-k notes by cosine similarity are returned (k=10 by default), and notes linked in the same box are also pulled in (Fig. 2 caption). Retrieval is one pass, with no agent loop.
- **Time**: a raw timestamp t_i on each note. No temporal indexing. **Summarization**: none beyond the per-note context X_i. **Learning**: none (prompting only).

**Handles changing facts/preferences?** Only indirectly: evolution rewrites neighbours' context and tags, but c_i is never overwritten. There is no explicit update or delete op for contradictions.

**Overlap**: (1) partial, since it uses LLM-enriched notes but not typed preference/fact/event records; (3) yes, vector search over notes that hold raw content. It has no SQL (2), no hierarchy (4) and no agentic retriever (5). The key difference is that A-MEM organizes memory as a linked graph of notes that evolve, while yours stores typed, dated records in tables alongside time-scaled summaries.

---

## MemoryOS (EMNLP, 2025) - `2025_emnlp_memoryos.pdf`

**Fundamental components**
1. **Three-tier storage**:
   - **STM**: a FIFO queue of "dialogue pages" {Q, R, timestamp T, meta_chain}, 7 pages long. The LLM links each page to the previous ones as a dialogue chain and summarizes the chain.
   - **MTM**: "segmented paging". Pages are grouped into topic segments when F_score = cos(embedding) + Jaccard(LLM keywords) > θ (0.6). The LLM summarizes each segment.
   - **LPM**: the User Persona (static profile; a User KB of extracted facts; User Traits over 90 LLM-updated dimensions) and the Agent Persona (profile and traits).
2. **STM->MTM update**: when the STM queue is full, the oldest page moves to MTM.
3. **Heat-based eviction and promotion**: Heat = α·N_visit + β·L_interaction + γ·exp(-Δt/μ). When MTM exceeds 2000 segments, the lowest-heat segments are evicted. Segments with heat > τ (5) update User Traits, User KB and Agent Traits, then their L_interaction resets. User KB and Agent Traits are FIFO queues capped at 100 entries.
4. **Retrieval**: all of STM; for MTM, top-m segments by F_score and then top-k pages by semantic similarity; for LPM, the top-10 KB and trait entries by semantic similarity plus the full profiles and User Traits. Everything is concatenated into one prompt, with no agent loop.
- **Time**: page timestamps plus a recency decay term inside heat. No calendar-level structure. **Learning**: none.

**Handles changing facts/preferences?** Partly: User Traits are "extracted and updated" by the LLM, and User KB entries fall out FIFO. There is no explicit contradiction or update op.

**Overlap**: (1) partial, since the LLM extracts user facts and traits into the KB but not as dated typed records; (3) yes, vector plus keyword retrieval over raw pages and segment summaries. It has no SQL (2) or agent retriever (5). Its tiers are organized by recency and heat, not calendar time (4). The key difference is that MemoryOS forgets by access frequency and recency, while yours keeps everything and compresses by time period.

---

## LightMem (ICLR, 2026) - `2026_iclr_lightmem.pdf`

**Fundamental components**
1. **Sensory memory**: LLMLingua-2 pre-compresses the raw turns by dropping low-retention tokens. A 512-token buffer is then cut into topic segments at boundaries where attention-matrix local maxima and low adjacent-turn similarity agree (Sec. 3.1).
2. **Topic-aware STM**: when the buffer reaches its token threshold, the LLM summarizes each topic group. The LTM entry is {topic, e_i = embedding(sum_i), user_i, model_i}, so the summary is stored together with the raw turns (Sec. 3.2). Fig. 2 shows typed items ([Identity]/[Activity]/[Preference]).
3. **LTM with soft updates**: at test time, new entries are simply inserted with timestamps.
4. **Sleep-time update**: offline, each entry gets an update queue of its top-k most similar entries, restricted to those with a later timestamp (t_j >= t_i). Updates then run in parallel, with ops {add, delete, update, merge} (Table 4) or ignore (Fig. 2), to "reorganize, de-duplicate, abstract… resolving inconsistencies".
5. **Retrieval**: cosine-similarity vector retrieval (all-MiniLM-L6-v2), with top entries added to the prompt. There is no agent loop. Fig. 2 mentions "global retrieval with time constraints" for the update step.
- **Time**: a timestamp per entry, and only later entries may update earlier ones. **Hierarchical summarization**: none beyond per-topic summaries. **Learning**: none.

**Handles changing facts/preferences?** Yes, through offline update, merge or delete, where a newer entry can overwrite an older one it conflicts with.

**Overlap**: (1) yes, LLM summaries/extraction into timestamped, partly typed entries; (3) yes, vector search over summaries that carry the raw turns. It has no SQL (2), no multi-level hierarchy (4) and no agent retriever (5). The key difference is that LightMem is built for efficiency (compression, batching, offline consolidation) and not for structured or time-scaled organization.

---

## Nemori (arXiv, 2025) - `2025_arxiv_nemori.pdf`

**Fundamental components**
1. **Episode partitioning**: messages (sender, content, timestamp) go into a buffer. At w=20 messages, the LLM splits the buffer into coherent raw episodes (Sec. 3.2.1).
2. **Narrative episodic memory**: for each episode, the LLM writes a third-person narrative N_j and a cue c_j. The prompt requires hour-precise absolute times and an ISO timestamp, resolving "yesterday" and similar references. Stored as M_j = (c_j, N_j, raw P_j, embedding v_j) in an episodic DB.
3. **Associative integration**: the top-K_e similar episodes are retrieved, and the LLM either merges the new episode into one of them (superseding it) or inserts it as new.
4. **Prediction-error semantic distillation**: the LLM predicts the episode from existing semantic knowledge alone. Whatever deviates from that prediction is extracted as semantic facts, so only unpredictable facts are stored. Consolidation against the top-K_m similar facts is one of new, merge (supersede) or conflict (purge outdated entries and replace). Results go to a semantic DB (Sec. 3.3).
5. **Retrieval**: runs in parallel, cosine top-k over episodes (k=10) and top-m over semantic facts (m=2k). The top-2 episodes also include their raw text, and everything is concatenated into the prompt. There is no agent loop.
- **Summarization**: episode-level narratives only, with no higher levels. **Learning**: none ("training-free").

**Handles changing facts/preferences?** Yes: the semantic "conflict" op purges outdated entries and replaces them, and "merge" supersedes them.

**Overlap**: (1) yes, LLM extraction of dated episodes and facts; (3) yes, vector search over narrative summaries plus raw episodes. It has no SQL (2), no multi-level hierarchy (4) and no agent retriever (5). The key difference is that Nemori decides what to store by prediction error and segments by episode, while yours extracts per day into a typed schema.

---

## Memory-R1 (ACL, 2026) - `2026_acl_memory-r1.pdf`

**Fundamental components**
1. **Extraction**: for each dialogue turn, LLMExtract produces key information/facts f_i (Alg. 3). The paper does not specify a typed schema.
2. **Memory bank**: a set of text memory entries per participant. The answer prompt says the memories "contain timestamped information". The storage backend is not specified in the paper.
3. **RL-trained Memory Manager**: it retrieves the top-K related memories and chooses {ADD, UPDATE (merge into the old entry), DELETE, NOOP} with new content m'. It is fine-tuned with PPO or GRPO, and the reward is the exact match of a frozen Answer Agent's answer. There are no operation labels, and training uses 152 LoCoMo QA pairs.
4. **Retrieval**: similarity-based RAG pulls 60 candidates (the top 30 per speaker).
5. **RL-trained Answer Agent**: it performs "memory distillation", selecting the useful memories and then answering. It is trained with PPO or GRPO on an EM reward. The agent does not make tool calls, since retrieval is a fixed step.
- **Time**: timestamps inside memory text, and the prompt tells the model to prefer the most recent memory when entries contradict. **Summarization**: none.

**Handles changing facts/preferences?** Yes, through learned UPDATE and DELETE operations, which overwrite or merge the entry.

**Overlap**: (1) yes, per-turn LLM fact extraction. There is no typed DB (2). (3) is partial, since it does vector RAG over extracted facts but not over raw chunks. It has no hierarchy (4). (5) is only loosely shared: the post-retrieval filter is learned, but it does not choose among stores. The key difference is that Memory-R1 learns memory operations with RL on a flat fact bank, while your pipeline is engineered and multi-store.

---

## Mem-α (arXiv, 2025) - `2025_arxiv_mem-alpha.pdf`

**Fundamental components**
1. **Three-part memory** (Sec. 3.3):
   - **Core**: one always-in-context summary paragraph of 512 tokens or less. Only `memory_update` (a full rewrite) applies.
   - **Semantic**: an expandable list of atomic factual statements.
   - **Episodic**: a chronological list of timestamped events.
2. **Tool-call writing**: for each incoming chunk, the agent issues a sequence of function calls `memory_insert`, `memory_update` or `memory_delete` (record id, memory type, content). Semantic and episodic memory support all three calls.
3. **RL training with GRPO** (Qwen3-4B, KL term dropped): the reward is r1 (QA accuracy of the final memory through RAG) + r2 (tool-call format success) + β·r3 (compression, 1 - memory length / input length) + γ·r4 (a Qwen3-32B judge of whether each op is semantically valid). Training uses 562 instances of 30K tokens or less, and the method generalizes to more than 400K tokens.
4. **Retrieval**: fixed and not learned. BM25 takes the top-k from semantic and episodic memory, core memory is always included, and a frozen LLM answers.
- **Time**: timestamps on episodic entries. **Summarization**: only the core paragraph, with no hierarchy. The storage backend is not specified in the paper beyond "lists".

**Handles changing facts/preferences?** The update and delete tools exist, but the authors explicitly exclude "Conflict Resolution" from training and evaluation. Whether it resolves conflicts in practice is not specified in the paper.

**Overlap**: (1) yes, extraction into facts (semantic) and dated events (episodic). There is no SQL (2). (3) is partial, since it uses BM25 over extracted entries, not vectors over raw chunks. (4) is minimal, a single core summary. (5) is inverted: the agent uses tools to write memory, not to retrieve it. The key difference is that Mem-α trains the writing policy with RL, while yours hand-designs the write step and makes retrieval agentic.
