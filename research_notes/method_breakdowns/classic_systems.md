# Classic / Reference Memory Systems: Method Breakdowns

Reference: the researcher's method components are
1) LLM extraction of preferences, facts and events per day (dated records),
2) SQL DB of records (entity, value, type, date),
3) RAG over raw chunks and summaries,
4) hierarchical summarization (weekly -> monthly -> yearly -> lifetime),
5) agentic tool-calling retriever that picks the store.

All statements below are taken from the PDFs in `papers/`. "Not specified in paper" means the paper does not say.

---

## MemoryBank (AAAI, 2024; arXiv May 2023)

**Fundamental components**
1. **Raw dialogue log with timestamps**: every multi-turn conversation is stored verbatim with timestamps, grouped by date ("In-Depth Memory Storage").
2. **Hierarchical event summary**: an LLM condenses each day's dialogue into a daily event summary ("Summarize the events and key information in the content"). The daily summaries are then combined into one global summary. There are two levels: daily and global.
3. **User portrait**: an LLM writes daily personality/emotion insights from each day's dialogue, then aggregates them into one global personality summary.
4. **Dense vector retrieval**: each conversation turn and each event summary is a memory piece. Pieces are encoded with a DPR-style dual-tower encoder (MiniLM for English, Text2vec for Chinese) and indexed in FAISS via LangChain. The current conversation context is the query.
5. **Ebbinghaus forgetting**: retention R = e^(-t/S), where t is the time since the memory was learned and S is its strength (starts at 1). Each recall adds 1 to S and resets t to 0. How low-retention items are actually removed is not specified in paper.

**Time representation**: timestamps on dialogue pieces and date-grouped daily summaries. Elapsed time t is used only for decay.

**Handles changing facts/preferences?** Only indirectly. The global portrait and summary are re-synthesized, and decay fades old, unrecalled items. There is no explicit contradiction/overwrite mechanism (not specified in paper).

**Overlap with researcher's method**: It shares (3) vector RAG over raw turns and summaries, and it partly shares (4), with daily summaries rolled into a global summary. It also has a weak form of (1): daily LLM summaries and personality notes, but not typed, dated fact records. It has no SQL store (2) and no agentic retriever (5), since retrieval is a single dense lookup. Its summary hierarchy has two levels, not a calendar hierarchy. It adds a forgetting curve, which the researcher's method lacks.

---

## Recursive Summarization, RSum / LLM-Rsum (Neurocomputing, 2025; arXiv 2308.15022)

**Fundamental components**
1. **Session-level memory iteration**: when a session S_i ends, an LLM rewrites the whole memory as M_i = LLM(S_i, M_{i-1}, P_m). The memory is a single natural-language block of at most 20 sentences about the user's and bot's "personality information". It starts as "none".
2. **Single flat memory**: only the latest memory M_N is kept. There is no tree, no database, no per-session archive, and older M_i are superseded (Markov chain P(M_i | S_i, M_{i-1})).
3. **Update through re-summarization**: the prompt tells the LLM to "identify any new or changed personality" and "combine the old and new personality information." The authors contrast this with earlier replace/append/delete "hard operations".
4. **No retrieval**: the response is r_t = LLM(C_t, M_N, P_r), with the whole memory placed in the prompt. The paper does show that RSum can be combined with BM25/DPR utterance retrieval (Section 6.6), but retrieval is not part of the method.
5. **Time representation**: not specified in paper. There are no dates or timestamps; order is implicit in the sequence of recursive updates.

**Handles changing facts/preferences?** Yes, implicitly. At each session end the LLM rewrites the memory and folds new or changed traits in (e.g., "recently joined a new gym"). Whether the old value is kept or dropped is left to the LLM.

**Overlap with researcher's method**: RSum is the precedent for the idea "re-summarize old summary plus new period" that underlies (4). It is not hierarchical, though. It has one level, updated per session, capped at 20 sentences, and it overwrites itself rather than keeping weekly, monthly and yearly layers. It does not keep dated structured records (1) or SQL (2). It has no RAG over chunks (3) and no agentic retriever (5), since the whole summary is always in context. The researcher's hierarchy adds period-bounded, dated summary layers that stay queryable. RSum keeps a single rolling, undated profile, so temporal questions ("what did I prefer last March?") cannot be answered once that fact is overwritten.

---

## MemTree (ICLR, 2025)

**Fundamental components**
1. **Unit of memory**: each new piece of information (in MSC, a dialogue round as the conversation unfolds; in QA, document chunks) becomes a node. A node holds text c_v, embedding e_v, parent, children and depth.
2. **Dynamic tree insertion**: starting at the (empty) root, the new node descends to the most similar child if cosine similarity is at least θ(d) = θ0·exp(λd) (θ0 = 0.4, rate 0.5). The threshold rises with depth. Otherwise the node is attached as a new leaf. A leaf that is reached is split into a parent with two children. The paper frames this as online hierarchical clustering (OTD) with an approximation bound.
3. **Recursive parent aggregation**: every ancestor on the insertion path is rewritten by an LLM Aggregate(c_v, c_new | n), which gets more abstract as the number of children n grows, and is then re-embedded.
4. **Collapsed-tree retrieval**: the query is compared with all nodes by cosine similarity, nodes below θ_retrieve are dropped, and the top-k are returned (k = 3 on MSC, k = 10 on MSC-E).
5. **Time representation / forgetting**: not specified in paper. There are no timestamps and no deletion.

**Handles changing facts/preferences?** Not addressed explicitly. New information is merged into ancestor summaries by LLM aggregation, while old leaves stay in the tree unchanged.

**Overlap with researcher's method**: It shares (3), embedding retrieval over raw leaves and summary nodes, and it has a form of hierarchical summarization (4). Its hierarchy is semantic (a similarity-clustered tree), not chronological (week, month, year). It has no dated fact extraction (1), no SQL (2) and no agentic retriever (5), since retrieval is one flat top-k similarity search.

---

## RAPTOR (ICLR, 2024)

**Fundamental components**
1. **Leaf chunks**: the corpus is split into contiguous chunks of about 100 tokens without breaking sentences, and each chunk is embedded with SBERT (multi-qa-mpnet-base-cos-v1).
2. **Recursive cluster-and-summarize tree**: leaf embeddings are reduced with UMAP and soft-clustered with GMMs, with the number of clusters chosen by BIC. Clustering is global first, then local. gpt-3.5-turbo summarizes each cluster, the summaries are re-embedded, and the loop repeats until clustering is no longer feasible. The result is a bottom-up tree built offline.
3. **Retrieval**: either tree traversal (top-k per layer, from the root down) or collapsed tree (cosine top-k over all nodes until 2,000 tokens). Collapsed tree is used because it performed better.
4. **Update/forgetting**: not specified in paper. The tree is built once over a static corpus, and there is no incremental insertion or deletion.
5. **Time representation**: not specified in paper. Clustering is by semantic similarity, "not just order in the text".

**Handles changing facts/preferences?** Not addressed. It is designed for static document QA (NarrativeQA, QASPER, QuALITY), not for evolving dialogue.

**Overlap with researcher's method**: It shares (3), vector RAG over chunks and summaries, and a semantic version of (4). It does not keep chronological, period-based layers, and it must be rebuilt rather than updated. It has no extraction (1), no SQL (2) and no agentic retriever (5).

---

## SeCom (ICLR, 2025)

**Fundamental components**
1. **Topical segmentation (unit of memory = segment)**: an LLM (GPT-4 zero-shot, or Mistral-7B / a fine-tuned RoBERTa) splits each session into topically coherent segments of consecutive turns. Optionally, the segmentation prompt is refined by LLM self-reflection on hard examples scored with WindowDiff.
2. **Flat segment-level memory bank**: segments are stored as raw dialogue text. There are no summaries, tree or database.
3. **Compression-based denoising**: LLMLingua-2 (xlm-roberta-large, 75% compression rate) removes redundant tokens from memory units before retrieval, R(u*, f_Comp(M), N).
4. **Retrieval**: BM25 or MPNet (multi-qa-mpnet-base-dot-v1) with FAISS, returning the top-N segments within a context budget. Retrieved units are put in the prompt "in time order".
5. **Update/forgetting**: not specified in paper. Memory only grows, and the memory bank is built from the history.

**Time representation**: only the session/turn order of retrieved units. No explicit dates.

**Handles changing facts/preferences?** Not specified in paper. There is no merge or overwrite; conflicting old and new segments may both be retrieved.

**Overlap with researcher's method**: It shares (3), RAG over raw conversation chunks, and its main contribution is better chunk granularity (topical segments rather than turns or sessions). It has no fact extraction (1), no SQL (2), no hierarchical summaries (4) and no agentic retriever (5). Its segmentation idea could replace the researcher's chunking step.

---

## Reflective Memory Management, RMM (ACL, 2025)

**Fundamental components**
1. **Prospective Reflection: topic-based extraction**: at the end of each session, an LLM breaks the dialogue into topic memories. Each memory is a pair of a "personal summary" (e.g., "SPEAKER_2 is considering joining a local gym") and the raw turns it references. The summary is the search key.
2. **LLM memory update (Add / Merge)**: for each extracted memory, the top-K similar existing memories are retrieved. An LLM then outputs either Add() for a new topic or Merge(index, merged_summary) when both discuss "the same aspect" of the user.
3. **Flat memory bank**: a list of (topic summary, raw dialogue) entries. There are no tables or tree.
4. **Dense retriever + learned reranker (Retrospective Reflection)**: a frozen retriever (Contriever by default, or Stella, or GTE) returns the top-K entries. A lightweight reranker (linear adapters with residual connections on query and memory embeddings, dot-product scores, Gumbel-softmax sampling) picks the top-M.
5. **Online RL from LLM citations**: the generator writes the response and cites the memories it used. Cited memories get +1 and uncited ones get -1, and the reranker is updated with REINFORCE.

**Time representation**: not specified in paper. There are no timestamps on memory entries.

**Handles changing facts/preferences?** Yes, through LLM Merge of the relevant topic summary. The merged text combines old and new (e.g., "exercises every Monday and Thursday, although he doesn't particularly enjoy it"). There is no explicit supersede-by-date.

**Overlap with researcher's method**: It shares (1) in part, since an LLM extracts per-session personal facts and preferences, though as free-text summaries without type or date. It also shares (3), dense retrieval over summaries linked to raw dialogue. It has no SQL store (2) and no hierarchical summaries (4). Its retrieval is adaptive but not agentic: a learned reranker rather than an LLM choosing which store to query (5).

---

## ChatDB (arXiv, 2023)

**Fundamental components**
1. **Database as symbolic memory**: a MySQL database with a task-specific relational schema. In the experiment this is a fruit shop with tables such as customers, suppliers, fruits, sales and sale_items, linked by primary and foreign keys. The schema is created up front, "manually or using LLMs".
2. **LLM controller writes records via SQL**: each incoming natural-language record (e.g., a purchase, sale, price change or return) is turned into INSERT, UPDATE or DELETE statements. Records are processed one by one in chronological order.
3. **Chain-of-Memory**: the LLM breaks the input into a sequence of SQL steps. Each step can be rewritten using the results of earlier SELECTs (e.g., filling a sale_id) before it runs. In-context exemplars and chain-of-thought are used.
4. **Retrieval = LLM-generated SQL SELECTs** (including calculations and joins) to answer questions, followed by an LLM "Response Summary". Input processing first decides whether memory is needed at all.
5. **Time representation**: dates are stored only where the schema has a date column (e.g., sales.sale_date). Attribute changes are done with in-place UPDATE (e.g., `UPDATE fruits SET selling_price = 1.6`), and the change date is not recorded. The paper says rollback "to any desired timestamp" is possible but does not describe how.

**Handles changing facts/preferences?** Yes, by SQL UPDATE or DELETE of the current value. This is exact state tracking, but it overwrites, so the history of a changed attribute is lost unless the schema logs it.

**Overlap with researcher's method**: It is the closest precedent for (2), and it partly anticipates (5), since the LLM decides whether to use memory and generates the queries. The key differences are these. ChatDB uses a domain-specific multi-table schema, not a generic (entity, value, type, date) record table. Its inputs are structured transaction records, not open-ended conversations, so there is no preference, fact or event extraction from dialogue (1). It overwrites values rather than appending dated versions. It has no vector RAG (3) and no summaries (4), and SQL is its only store, so there is no choice between stores. It was evaluated only on a synthetic 70-record fruit-shop dataset with 50 questions.

---

## Quick comparison vs researcher's components

| System | (1) dated extraction | (2) SQL | (3) vector RAG | (4) hierarchical summaries | (5) agentic multi-store retriever |
|---|---|---|---|---|---|
| MemoryBank | partial (daily summaries/portrait, not typed records) | no | yes (turns + summaries, FAISS) | partial (daily -> global) | no |
| RSum | no | no | no (whole memory in prompt) | partial (single rolling recursive summary, per session) | no |
| MemTree | no | no | yes (collapsed tree) | semantic tree, not temporal | no |
| RAPTOR | no | no | yes (collapsed tree) | semantic tree, offline, static | no |
| SeCom | no | no | yes (segments, BM25/MPNet) | no | no |
| RMM | partial (topic summaries, undated) | no | yes (+ RL reranker) | no | no (learned reranker) |
| ChatDB | no (structured inputs; SQL writes) | yes (task schema, UPDATE in place) | no | no | partial (LLM-generated SQL chain over one store) |
