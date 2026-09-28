# Long-term memory frameworks for LLM agents (as of September 2026): baseline profiles

Scope: Mem0 / Mem0^g, Zep / Graphiti, MemGPT / Letta, LangMem, MemOS, MIRIX, Cognee, Memobase, and other widely used systems found during research (A-MEM, MemoryOS, LightMem, Hindsight, EverMemOS/EverOS, MemMachine, Supermemory, Memori). Research date: 2026-09-28. Star counts are snapshots taken from GitHub pages on that date.

## Q1. Associated paper, authors, year, venue (peer-reviewed or arXiv-only), and PDF link

### Takeaway
Among the requested frameworks, only **Mem0 (ECAI 2025)** has a verified peer-reviewed venue. Zep, MemGPT, MemOS, MIRIX and Cognee are arXiv-only, and LangMem and Memobase have no paper. Several of the newer systems do have strong venues: **A-MEM (NeurIPS 2025)**, **MemoryOS (EMNLP 2025, main track, oral)** and **LightMem (ICLR 2026)**.

### Cited Findings

**Summary table** (venue checked against proceedings, ACL Anthology, OpenReview, arXiv or DBLP where noted)

| Framework | Paper | Authors | Year | Venue (verified) | PDF |
|---|---|---|---|---|---|
| Mem0 / Mem0^g | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | Chhikara, Khant, Aryan, Singh, Yadav (Mem0) | 2025 | **ECAI 2025**, FAIA vol. 413, pp. 2993-3000, DOI 10.3233/FAIA251160 | https://arxiv.org/pdf/2504.19413 |
| Zep / Graphiti | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | Rasmussen, Paliychuk, Beauvais, Ryan, Chalef (Zep AI) | Jan 2025 | arXiv-only (2501.13956) | https://arxiv.org/pdf/2501.13956 |
| MemGPT / Letta | MemGPT: Towards LLMs as Operating Systems | Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez (UC Berkeley) | Oct 2023 (**older than 2024**), v2 Feb 2024 | arXiv-only (DBLP lists only CoRR) | https://arxiv.org/pdf/2310.08560 |
| LangMem | none (library and docs only) | LangChain | 2025 | n/a | n/a |
| MemOS | MemOS: A Memory OS for AI System | Zhiyu Li, Chenyang Xi, Chunyu Li + 36 others (MemTensor et al.) | Jul 2025 (rev. Dec 2025) | arXiv-only (2507.03724); earlier short version 2505.22101 | https://arxiv.org/pdf/2507.03724 |
| MIRIX | MIRIX: Multi-Agent Memory System for LLM-Based Agents | Yu Wang, Xi Chen | Jul 2025 | arXiv-only (2507.07957) | https://arxiv.org/pdf/2507.07957 |
| Cognee | Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning | Markovic, Obradovic, Hajdu, Pavlovic | May 2025 | arXiv-only (2505.24478), self-described as a "preliminary version" | https://arxiv.org/pdf/2505.24478 |
| Memobase | none | memodb-io | n/a | n/a | n/a |
| A-MEM | A-Mem: Agentic Memory for LLM Agents | Xu, Liang, Mei, Gao, Tan, Zhang | 2025 | **NeurIPS 2025** (main conference, poster) | https://arxiv.org/pdf/2502.12110 |
| MemoryOS | Memory OS of AI Agent | Kang, Ji, Zhao, Bai (BAI-LAB) | 2025 | **EMNLP 2025 main** (2025.emnlp-main.1318; README says Oral) | https://aclanthology.org/2025.emnlp-main.1318.pdf |
| LightMem | LightMem: Lightweight and Efficient Memory-Augmented Generation | zjunlp (Zhejiang Univ. NLP) | Oct 2025 | **ICLR 2026** (poster) | https://arxiv.org/pdf/2510.18866 |
| Hindsight | Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects | Latimer, Boschi, Neeser, Bartholomew, Srivastava, Wang, Ramakrishnan (Vectorize, Virginia Tech) | Dec 2025 | arXiv-only (2512.12818) | https://arxiv.org/pdf/2512.12818 |
| EverMemOS (now EverOS) | EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning | Chuanrui Hu, ..., Lidong Bing, Yafeng Deng | Jan 2026 | arXiv-only (2601.02163) | https://arxiv.org/pdf/2601.02163 |
| MemMachine | MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents | Shu Wang, ..., Charles Fan | Apr 2026 | arXiv-only (2604.04853) | https://arxiv.org/pdf/2604.04853 |
| Supermemory | no paper found; company "research" pages only | Supermemory | n/a | n/a | n/a |

**Sources for the table rows**
- Mem0 is in the ECAI 2025 proceedings: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory", Frontiers in AI and Applications vol. 413, pp. 2993-3000, DOI 10.3233/FAIA251160 — [IOS Press](https://ebooks.iospress.nl/doi/10.3233/FAIA251160). The arXiv v1 was posted 28 Apr 2025 — [arXiv](https://arxiv.org/abs/2504.19413)
- Zep paper, submitted 20 Jan 2025. No venue is listed — [arXiv](https://arxiv.org/abs/2501.13956)
- MemGPT authors are Packer, Wooders, Lin, Fang, Patil, Stoica and Gonzalez (UC Berkeley) — [arXiv PDF](https://arxiv.org/pdf/2310.08560). DBLP lists the paper only as a CoRR entry — [dblp](https://dblp.org/rec/journals/corr/abs-2310-08560.html). The v1 was posted 12 Oct 2023 and the v2 on 12 Feb 2024 — [arXiv](https://export.arxiv.org/abs/2310.08560v2)
- MemOS arXiv 2507.03724: submitted 4 Jul 2025, revised 3 Dec 2025, 39 authors — [arXiv](https://arxiv.org/abs/2507.03724). The repo cites both 2507.03724 and 2505.22101 — [GitHub](https://github.com/MemTensor/MemOS)
- MIRIX arXiv 2507.07957 by Yu Wang and Xi Chen, 10 Jul 2025 — [arXiv](https://arxiv.org/abs/2507.07957)
- Cognee paper arXiv 2505.24478, 30 May 2025, described as a "preliminary version with a revised, expanded version in preparation" — [arXiv](https://arxiv.org/abs/2505.24478). The Cognee README cites this paper — [GitHub](https://github.com/topoteretes/cognee)
- Memobase has no academic paper linked — [GitHub](https://github.com/memodb-io/memobase)
- LangMem: no paper is mentioned in its docs or repo — [docs](https://langchain-ai.github.io/langmem/), [GitHub](https://github.com/langchain-ai/langmem)
- A-MEM is in the NeurIPS 2025 proceedings — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119020), [arXiv](https://arxiv.org/abs/2502.12110)
- MemoryOS is in EMNLP 2025 main — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1318/). The repo is tagged "[EMNLP 2025 Oral]" — [GitHub](https://github.com/BAI-LAB/MemoryOS)
- LightMem is in the ICLR 2026 proceedings — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a05b72653ec5b473732129829ae04195-Abstract-Conference.html), [OpenReview](https://openreview.net/forum?id=dyJ0GWpjJB), [arXiv](https://arxiv.org/abs/2510.18866)
- Hindsight: submitted 14 Dec 2025, no venue — [arXiv](https://arxiv.org/abs/2512.12818)
- EverMemOS: v1 5 Jan 2026 and v2 9 Jan 2026. The comments field lists pages only, with no venue — [arXiv](https://arxiv.org/abs/2601.02163)
- MemMachine: 6 Apr 2026, no venue — [arXiv](https://arxiv.org/abs/2604.04853)
- Benchmark papers, for context (both are older than this survey's focus, but they are the standard benchmarks): LongMemEval is referred to as "(ICLR 2025)" in the MemMachine abstract — [arXiv](https://arxiv.org/abs/2604.04853)

### Inferences
- If a reviewer insists on "published baselines", the defensible choices are Mem0 (ECAI 2025), A-MEM (NeurIPS 2025), MemoryOS (EMNLP 2025) and LightMem (ICLR 2026). Zep and MemGPT are heavily cited but arXiv-only.
- Industry vendors (Zep, Letta, Supermemory, Memobase, EverMind, Vectorize) publish most of their numbers on blogs. Treat those numbers as self-reported.

### Gaps
- I could not confirm whether MemGPT was ever submitted to and rejected from ICLR 2024 on OpenReview. The only verified fact is that DBLP lists only the CoRR version.
- I found no evidence that MemOS, MIRIX, Zep, Cognee, Hindsight, EverMemOS or MemMachine were accepted at a peer-reviewed venue as of Sep 2026. The search was not exhaustive for ACL 2026, KDD 2026 or AAAI 2026 accepted-paper lists.
- I did not fetch the full author list for LightMem.

## Q2. Core idea: how memory is extracted, stored, updated and retrieved, and whether temporal validity is handled

### Takeaway
There are four design families:
- **LLM fact extraction with CRUD updates in a vector store:** Mem0, LangMem, Memobase profiles.
- **Temporal knowledge graphs with explicit validity intervals:** Zep/Graphiti, and partly Mem0^g.
- **Agent-managed tiered context, OS style:** MemGPT/Letta, MemOS, MemoryOS, MIRIX.
- **Structured multi-network or episodic consolidation:** Hindsight, EverMemOS, A-MEM, LightMem.

Graphiti has the most explicit handling of fact validity over time: bi-temporal edges that are invalidated rather than deleted.

### Cited Findings

**Mem0**
- Mem0 processes message pairs in two phases. In the extraction phase, an LLM pulls salient memories using a conversation summary plus the m=10 most recent messages. In the update phase, it compares each candidate with the s=10 most similar stored memories and applies one of ADD, UPDATE, DELETE or NOOP — [arXiv HTML](https://arxiv.org/html/2504.19413)
- The experiments used GPT-4o-mini for extraction and OpenAI text-embedding-3-small for embeddings — [arXiv HTML](https://arxiv.org/html/2504.19413)
- **Mem0^g** stores memories in Neo4j as a directed labeled graph (entity nodes, relation edges). Conflicting relationships are marked invalid rather than deleted, which supports temporal reasoning — [arXiv HTML](https://arxiv.org/html/2504.19413)
- In April 2026 Mem0 released a new algorithm with "single-pass extraction, entity linking, multi-signal retrieval, and temporal reasoning". The README says the hosted platform has "proprietary optimizations not available in the open-source SDK" — [GitHub](https://github.com/mem0ai/mem0)
- The current defaults are gpt-5-mini as the LLM and text-embedding-3-small for embeddings — [GitHub](https://github.com/mem0ai/mem0)

**Zep / Graphiti**
- Graphiti is a "temporally-aware knowledge graph engine" that ingests both conversational and structured business data while preserving historical relationships — [arXiv](https://arxiv.org/abs/2501.13956)
- It uses a bi-temporal model: timeline T is the chronological order of events and T' is the order in which data was ingested — [arXiv HTML](https://arxiv.org/html/2501.13956)
- Facts carry validity windows. Outdated facts are invalidated, not deleted — [GitHub](https://github.com/getzep/graphiti)
- In the paper, BGE-m3 was used for both embedding and reranking — [arXiv HTML](https://arxiv.org/html/2501.13956)

**MemGPT / Letta**
- MemGPT manages memory in tiers, analogous to an operating system. The LLM moves data between main context and external storage (archival and recall) through function calls, and interrupts manage control flow — [arXiv PDF](https://arxiv.org/pdf/2310.08560); [Semantic Scholar](https://www.semanticscholar.org/paper/MemGPT:-Towards-LLMs-as-Operating-Systems-Packer-Fang/908dad62c0e43d80e3e3cb3c0402f7c71c70499c)
- Letta continues the MemGPT line: main context acts as RAM, archival memory as disk, and the agent manages memory itself through tools — [Vectorize comparison](https://vectorize.io/articles/best-ai-agent-memory-systems)
- Letta now also has a "git-backed memory filesystem" (`LETTA_MEMFS_SERVICE_URL`) — [Letta docs](https://docs.letta.com/guides/selfhosting)

**LangMem**
- Offers two modes: "hot path" tools the agent calls itself (`create_manage_memory_tool`, `create_search_memory_tool`), and a background memory manager that extracts, consolidates and updates memories automatically. Memory types are semantic and episodic, plus procedural (prompt optimization, per LangChain) — [docs](https://langchain-ai.github.io/langmem/)
- Storage is LangGraph's BaseStore (InMemoryStore, AsyncPostgresStore) — [docs](https://langchain-ai.github.io/langmem/)

**MemOS**
- The core unit is the "MemCube", which wraps memory content together with metadata such as provenance and versioning. MemOS unifies three kinds of memory: plaintext, activation-based (KV-cache) and parameter-level (e.g., LoRA). MemCubes can be composed, migrated and fused — [arXiv](https://arxiv.org/abs/2507.03724)

**MIRIX**
- Has six memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault), each managed by a dedicated agent under multi-agent coordination. It is multimodal, for example screen-capture memory — [arXiv](https://arxiv.org/abs/2507.07957)
- Retrieval uses PostgreSQL-native BM25 plus embeddings — [GitHub](https://github.com/Mirix-AI/MIRIX)

**Cognee**
- Runs an Extract-Cognify-Load pipeline that turns documents, code and conversations into a typed knowledge graph plus vector index — [Vectorize comparison](https://vectorize.io/articles/best-ai-agent-memory-systems); [GitHub](https://github.com/topoteretes/cognee)
- The paper is about hyperparameter tuning of the knowledge-graph/RAG pipeline (chunking, graph construction, retrieval, prompting). It is not about conversational memory as such — [arXiv](https://arxiv.org/abs/2505.24478)

**Memobase**
- Keeps a structured user profile (basic info, interests, psychology, work, ...) plus a timestamped event timeline per user, and targets under 100 ms retrieval — [GitHub](https://github.com/memodb-io/memobase)
- Since v0.0.40 it uses a fixed 3 LLM calls per flush, down from roughly 3-10 — [GitHub](https://github.com/memodb-io/memobase)

**A-MEM**
- Zettelkasten-style: each new memory becomes a note with a contextual description, keywords and tags, and is dynamically linked to related notes. Existing notes "evolve" as new ones arrive — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html)

**MemoryOS**
- Hierarchical short-, mid- and long-term storage with four modules (Storage, Updating, Retrieval, Generation), inspired by OS memory management — [GitHub](https://github.com/BAI-LAB/MemoryOS)

**LightMem**
- Follows the three-stage Atkinson-Shiffrin model. A sensory stage filters and compresses input, topic-aware short-term memory groups and summarizes it, and consolidation into long-term memory happens offline "sleep-time" — [search summary of repo/ICLR page](https://github.com/zjunlp/LightMem). The details of the long-term stage were not verified from the paper text.

**Hindsight**
- Uses four logical memory networks: world facts, agent experiences, synthesized entity summaries, and evolving beliefs. It exposes three operations: retain, recall and reflect, and includes temporal and entity-aware layers — [arXiv](https://arxiv.org/abs/2512.12818)
- Retrieval runs semantic, BM25, graph and temporal search in parallel, then reranks with a cross-encoder — [Vectorize](https://vectorize.io/articles/best-ai-agent-memory-systems) (vendor source)

**EverMemOS**
- Has three stages, described as "engram-inspired":
  1. Dialogue is turned into MemCells (episodic trace, atomic facts, and time-bounded "Foresight" signals).
  2. MemCells are consolidated into thematic MemScenes, and the user profile is updated.
  3. Retrieval is agentic and guided by MemScenes.
- It explicitly targets consolidating evolving user state and resolving conflicts — [arXiv](https://arxiv.org/abs/2601.02163)

**MemMachine**
- Has short-term, long-term episodic and profile memory layers. It stores raw episodes ("ground-truth-preserving") and retrieves with contextualized expansion around matched turns. The authors found that retrieval-stage changes mattered more than ingestion-stage changes — [arXiv](https://arxiv.org/abs/2604.04853)

**Supermemory**
- Stores atomic memories with "relational versioning", temporal metadata and source chunks. It handles knowledge updates and contradictions and forgets expired information — [Supermemory research](https://supermemory.ai/research/longmembench/) (vendor source)

### Inferences
- For a lifelong-memory method centred on fact updates and temporal validity:
  - Graphiti (explicit valid/invalid intervals) and Mem0 (explicit UPDATE/DELETE, with graph invalidation in Mem0^g) are the most directly comparable baselines.
  - MemGPT/Letta is the canonical agent-managed baseline.
  - A-MEM and MemoryOS are the canonical academic baselines.
- LangMem and Memobase are mainly engineering systems with no paper. They are useful as practical baselines but have no peer-reviewed method description to cite.

### Gaps
- The details of Mem0's April 2026 algorithm have not been published as a paper. The only source is the blog and README.
- I did not verify from the LightMem paper text its long-term update mechanism or whether it handles temporal validity.

## Q3. Reported benchmarks, headline numbers (metric and judge), and disputed results

### Takeaway
**LoCoMo with an LLM-as-judge "J" score** is the de facto shared benchmark, but results are not comparable across papers. Judge prompts, answer models, category filtering (whether adversarial category 5 is excluded) and system configuration all differ.

The Mem0 vs Zep dispute is the clearest example. For the same system, Zep, reported scores range from 58.44% to 84%.

**LongMemEval** is increasingly preferred (Zep, Hindsight, MemMachine, Supermemory, MemOS). **BEAM** appeared in 2026 (Mem0, Cognee).

Headline self-reported numbers:

| System | LoCoMo | LongMemEval |
|---|---|---|
| Mem0 (paper) | 66.88 / 68.44 (Mem0^g) | not reported |
| Mem0 (Apr 2026 algorithm) | 92.5 | 94.4 |
| Zep (paper) | not reported | 71.2 (gpt-4o) |
| MemOS | 75.80 (paper), 88.83 (README) | not listed above |
| MIRIX | 85.4 | not listed above |
| Memobase | 75.78 | not listed above |
| Hindsight | 89.61 | 91.4 |
| MemMachine | 91.69 | 93.0 |
| Letta filesystem | 74.0 | not reported |

### Cited Findings

**Mem0 paper (ECAI 2025)**
- Setup: LoCoMo categories 1-4, LLM-as-a-Judge (J) with GPT-4o-mini at temperature 0 as the judge, and a GPT-4o-mini backbone — [arXiv HTML](https://arxiv.org/html/2504.19413)
- Overall J scores:

  | System | Overall J |
  |---|---|
  | Full-context | 72.90 |
  | Mem0^g | 68.44 |
  | Mem0 | 66.88 |
  | Zep | 65.99 |
  | LangMem | 58.10 |
  | OpenAI memory | 52.90 |

  — [arXiv HTML](https://arxiv.org/html/2504.19413)
- Latency (p95 total): Mem0 1.44 s, Mem0^g 2.59 s, Zep 2.93 s, full-context 17.12 s — [arXiv HTML](https://arxiv.org/html/2504.19413)
- Memory footprint: Mem0 about 7k tokens and Mem0^g about 14k tokens per conversation, versus more than 600k for Zep — [arXiv HTML](https://arxiv.org/html/2504.19413)
- Mem0's own reported numbers put full-context above Mem0 on J. Its headline claim, "26% relative improvement", is relative to OpenAI memory — [arXiv](https://arxiv.org/abs/2504.19413)

**Mem0, April 2026 algorithm (self-reported)**
- LoCoMo 92.5, LongMemEval 94.4, BEAM-1M 64.1, BEAM-10M 48.6, each at about 7k tokens per query — [Mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- The evaluation harness is open-sourced at github.com/mem0ai/memory-benchmarks — [Mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- The README says open-source users should expect "directionally similar gains but not identical numbers" — [GitHub](https://github.com/mem0ai/mem0)

**Zep paper**

| Benchmark | Backbone | Zep | Comparison |
|---|---|---|---|
| DMR | gpt-4-turbo | 94.8% | MemGPT 93.4% |
| DMR | gpt-4o-mini | 98.2% | |
| DMR | | | recursive summarization 35.3% |
| LongMemEval | gpt-4o-mini | 63.8% | full-context 55.4% |
| LongMemEval | gpt-4o | 71.2% | full-context 60.2% (+18.5%) |

- For LongMemEval, latency was about 90% lower (2.58 s vs 28.9 s) and context was about 1.6k tokens vs 115k — [arXiv HTML](https://arxiv.org/html/2501.13956)
- The judge was GPT-4o, using LongMemEval's question-specific prompts — [arXiv HTML](https://arxiv.org/html/2501.13956)
- Zep scored worse than full-context on the single-session-assistant question type: -17.7% with gpt-4o and -9.06% with gpt-4o-mini — [arXiv HTML](https://arxiv.org/html/2501.13956)
- Note: one third-party blog attributes Zep's 63.8% to GPT-4o. That is wrong; per the paper, 63.8% is gpt-4o-mini — [Vectorize](https://vectorize.io/articles/best-ai-agent-memory-systems) vs [arXiv HTML](https://arxiv.org/html/2501.13956)

**The Mem0 vs Zep LoCoMo dispute (2025)**
1. **Zep's claim.** Zep published a LoCoMo score of 84%.
2. **Mem0's challenge, 8 May 2025.** Mem0 CTO Deshraj Yadav filed a GitHub issue saying Zep had counted adversarial category-5 answers in the numerator but not the denominator, inflating the score by about 25.56 points. He also cited modified prompts and single runs, and gave a corrected score of 58.44% ± 0.20 — [getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5)
3. **Zep's rebuttal, blog of 6 May 2025 (updated 3 Jun 2026).** Zep said Mem0's own evaluation of Zep was misconfigured in three ways: it assigned the user role to both speakers, appended timestamps to message text instead of using `created_at`, and ran searches sequentially. Zep reported a corrected score of 75.14% ± 0.17 J, which is about 10% above Mem0^g's roughly 68%, with p95 search latency of 0.632 s — [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
4. **Zep's critique of LoCoMo itself.** Conversations are only about 16k-26k tokens, so they fit in context. There are no knowledge-update questions. There are missing or incorrect ground truths and speaker attribution errors. And full-context (about 73%) beats Mem0 (about 68%) — [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
5. **Related user report.** A user filed an issue about inconsistencies between the Mem0 paper's Zep numbers and Zep's own — [getzep/zep#405](https://github.com/getzep/zep/issues/405)
6. **Later claims.**
   - Mem0's 2026 blog lists Zep at 80.32% on LoCoMo — [Mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
   - Third-party aggregators cite Zep claims of 94.7% on LoCoMo and describe the setups as "not directly comparable" — [Dakera](https://dakera.ai/benchmark/); [Mem0 benchmark guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
   - Memori Labs reports Zep 79.09, LangMem 78.05 and Mem0 62.47 on its own harness — [Memori docs](https://memorilabs.ai/docs/memori-cloud/benchmark/results/)

**Letta and MemGPT**
- Letta's "filesystem" agent (gpt-4o-mini, conversation history stored in files) scored 74.0% on LoCoMo, above Mem0^g's 68.5%. Letta argues the benchmark mostly measures context management, not the memory mechanism (blog, Aug 2025) — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory/)
- MemGPT paper, 2023: on DMR (based on MSC), accuracy / ROUGE-L were:

  | Backbone | Baseline | + MemGPT |
  |---|---|---|
  | GPT-3.5 Turbo | 38.7% / 0.394 | 66.9% / 0.629 |
  | GPT-4 | 32.1% / 0.296 | 92.5% / 0.814 |
  | GPT-4 Turbo | 35.3% / 0.359 | 93.4% / 0.827 |

  The paper also reports a conversation-opener task (SIM-1/3/H), document QA and nested key-value retrieval — [arXiv PDF](https://arxiv.org/pdf/2310.08560)

**MemOS (paper v4, all methods on a GPT-4o-mini backbone)**
- LoCoMo LLM-judge overall scores:

  | System | Overall |
  |---|---|
  | MemOS-1031 | 75.80 |
  | Memobase | 72.01 |
  | Mem0 | 64.57 |
  | MIRIX | 64.33 |
  | Zep | 59.22 |
  | MemU | 56.55 |
  | Supermemory | 55.34 |

  — [arXiv PDF](https://arxiv.org/pdf/2507.03724)
- LongMemEval overall scores:

  | System | Overall |
  |---|---|
  | MemOS | 77.8 |
  | Memobase | 72.4 |
  | Mem0 | 66.4 |
  | Zep | 63.8 |
  | Supermemory | 58.4 |
  | MIRIX | 43.49 |
  | MemU | 38.4 |

  — [arXiv PDF](https://arxiv.org/pdf/2507.03724)
- The paper also reports PrefEval and PersonaMem — [arXiv PDF](https://arxiv.org/pdf/2507.03724)
- The README reports newer numbers on "OmniMemEval": LoCoMo 88.83, LongMemEval 89.20, PersonaMem v2 40.58, HaluMem 80.91. No baseline names are given for these — [GitHub](https://github.com/MemTensor/MemOS)

**MIRIX**
- LoCoMo 85.4%, claimed SOTA. ScreenshotVQA accuracy 35% higher than RAG with 99.9% less storage — [arXiv](https://arxiv.org/abs/2507.07957)
- MemOS's own re-run of MIRIX scored only 64.33 on LoCoMo. That is a large discrepancy between self-reported and third-party numbers — [arXiv PDF](https://arxiv.org/pdf/2507.03724)

**Memobase (self-reported, LoCoMo, judge "e.g. gpt-4o")**
- v0.0.37 scored 75.78 overall (temporal 85.05) and v0.0.32 scored 70.91. The baseline numbers are copied from the Mem0 paper — [GitHub docs](https://github.com/memodb-io/memobase/tree/main/docs/experiments/locomo-benchmark)

**Cognee**
- The paper uses HotPotQA, 2WikiMultiHop and MuSiQue with EM, F1 and a DeepEval LLM correctness metric. It does not use a conversational-memory benchmark — [arXiv](https://arxiv.org/abs/2505.24478)
- The README reports BEAM scores of 0.79 (100K) and 0.67 (10M), and notes these use benchmark-specific formatting — [GitHub](https://github.com/topoteretes/cognee)

**MemoryOS**
- On LoCoMo with GPT-4o-mini, average improvements over baselines of +48.36% F1 and +46.18% BLEU-1. It does not use an LLM-judge metric — [GitHub](https://github.com/BAI-LAB/MemoryOS)

**Hindsight**
- LongMemEval 83.6% with an open 20B model and 91.4% with a scaled backbone, against a 39% baseline. LoCoMo 89.61% vs 75.78% for the "previous strongest open system" (that is Memobase's number) — [arXiv](https://arxiv.org/abs/2512.12818)

**MemMachine**
- LoCoMo 0.9169 (the abstract says "gpt-4-mini", presumably gpt-4.1-mini), LongMemEval-S 93.0%. It claims about 80% fewer input tokens than Mem0 — [arXiv](https://arxiv.org/abs/2604.04853)

**Supermemory**
- LongMemEval 95% Recall@15 with about 720 context tokens. This is a retrieval-recall metric, not QA accuracy (self-reported) — [Supermemory research](https://supermemory.ai/research/longmembench/)

### Inferences
- A baseline table taken from any vendor paper will be internally inconsistent. The same system appears with very different scores:
  - Zep on LoCoMo: 58.44, 59.22, 65.99, 75.14, 80.32, 84 and 94.7.
  - Mem0 on LoCoMo: 62.47, 64.57, 66.88 and 92.5.

  The safest approach for a new method is to re-run every baseline in one harness, with the same answer LLM, judge model and judge prompt, over several runs, and with category 5 excluded. Then report full-context and naive-RAG controls alongside.
- LoCoMo is saturated and its conversations fit in context, so it does not test knowledge updates well (this is Zep's critique). For lifelong memory with fact updates, LongMemEval (its knowledge-update and temporal-reasoning types) is more diagnostic, and so is BEAM for longer horizons.

### Gaps
- The judge model for the MemOS LoCoMo and LongMemEval tables was not confirmed. Only the GPT-4o-mini backbone was confirmed.
- I did not extract A-MEM's and LightMem's own headline numbers from their papers. A-MEM's arXiv version reports LoCoMo F1 and BLEU; the LightMem abstract claims accuracy gains and large token reductions on LongMemEval. Both need verification.
- I found no primary Zep publication for the 94.7% LoCoMo figure. It is known only through third-party aggregators.

## Q4. Open-source code: repo, license, activity, Gemini/Vertex support, and effort to run on a custom daily-conversation dataset

### Takeaway
Every framework profiled has an Apache-2.0 or MIT repo.

Gemini support:
- **First-class Gemini and Vertex support** for both LLM and embeddings: Mem0 (Google AI and Vertex AI embedders; Google AI and LiteLLM for the LLM), Graphiti (LLM, embedder and reranker), Cognee (Gemini through AI Studio or Vertex), Letta (Gemini and Vertex).
- **Gemini is the default:** MIRIX (Gemini 2.0 Flash with text-embedding-004).

Lightest to run on a custom daily-conversation dataset:
- Mem0 OSS (a vector store only; graph optional).
- LangMem (in-memory or Postgres store).
- EverOS (SQLite + LanceDB).

Heavier:
- Graphiti (needs Neo4j or FalkorDB).
- MemOS (Neo4j + Qdrant).
- Letta (Postgres + pgvector server).
- Memobase (Postgres + Redis).

Zep's managed "Context Graph Engine" and Mem0's April 2026 platform optimizations are hosted-only. Paper numbers for those products cannot be exactly reproduced with the OSS code.

### Cited Findings

**Mem0**
- https://github.com/mem0ai/mem0, Apache-2.0, about 66.2k stars — [GitHub](https://github.com/mem0ai/mem0). The blog says 62,590 stars at an earlier date — [Mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- LLM providers include Google AI, LiteLLM, OpenAI, Anthropic, Ollama and others — [Mem0 docs](https://docs.mem0.ai/components/llms/overview)
- Embedders include Google AI and Vertex AI — [Mem0 docs](https://docs.mem0.ai/components/embedders/overview)
- Some platform optimizations are proprietary — [GitHub](https://github.com/mem0ai/mem0)

**Graphiti / Zep**
- https://github.com/getzep/graphiti, Apache-2.0, about 31.3k stars, about 994 commits — [GitHub](https://github.com/getzep/graphiti)
- Requires a graph database: Neo4j 5.26+, FalkorDB 1.1.2+ or Amazon Neptune. Kuzu support is deprecated — [GitHub](https://github.com/getzep/graphiti)
- Gemini is available through the `google-genai` extra, with `GeminiClient` (LLM), `GeminiEmbedder` and `GeminiRerankerClient` (gemini-2.5-flash-lite). Anthropic, Groq, Azure and OpenAI-compatible endpoints are also supported — [GitHub](https://github.com/getzep/graphiti)
- Zep Cloud, with its proprietary "Context Graph Engine", user and conversation management, and sub-200 ms retrieval, is hosted-only — [GitHub](https://github.com/getzep/graphiti)

**Letta**
- https://github.com/letta-ai/letta, Apache-2.0, about 25k stars. The repo front page now centres on Letta Code (npm, desktop app, cloud) — [GitHub](https://github.com/letta-ai/letta)
- Self-hosting needs PostgreSQL with pgvector. The Docker image `letta/letta` is noted as "no longer actively maintained" — [Letta docs](https://docs.letta.com/guides/selfhosting)
- An embedding model must be specified when creating agents on a self-hosted server — [Letta docs](https://docs.letta.com/guides/selfhosting)
- Google Gemini (AI Studio key) and Google Vertex AI (local only) are supported providers — [Letta docs](https://docs.letta.com/guides/server/providers/google)

**LangMem**
- https://github.com/langchain-ai/langmem, MIT, about 1.7k stars, 153 commits, `pip install -U langmem` — [GitHub](https://github.com/langchain-ai/langmem)
- Storage is LangGraph BaseStore (InMemoryStore or AsyncPostgresStore) — [docs](https://langchain-ai.github.io/langmem/)

**MemOS**
- https://github.com/MemTensor/MemOS, Apache-2.0, about 11.6k stars, 2,177 commits — [GitHub](https://github.com/MemTensor/MemOS)
- Self-hosting requires Neo4j and Qdrant, with Redis optional, and Docker Compose is provided. A cloud API exists at memos.memtensor.cn — [GitHub](https://github.com/MemTensor/MemOS)

**MIRIX**
- https://github.com/Mirix-AI/MIRIX, Apache-2.0, about 3.4k stars, 383 commits — [GitHub](https://github.com/Mirix-AI/MIRIX)
- Requires PostgreSQL. Runs via Docker Compose or `pip install mirix-client`. Examples use Gemini 2.0 Flash with Google text-embedding-004 (768-dimensional) — [GitHub](https://github.com/Mirix-AI/MIRIX)

**Cognee**
- https://github.com/topoteretes/cognee, Apache-2.0, about 31.1k stars, v1.6.1 released 24 Sep 2026, so actively maintained — [GitHub](https://github.com/topoteretes/cognee)
- Default local storage is SQLite + LanceDB + Kuzu, with Neo4j and Postgres as options — [GitHub](https://github.com/topoteretes/cognee)
- Gemini is configured with `LLM_PROVIDER="gemini"`, `LLM_MODEL="gemini/gemini-2.0-flash"`, or `vertex_ai/...` with `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION`. If only the LLM or only the embedder is configured, the other defaults to OpenAI — [Cognee docs](https://docs.cognee.ai/setup-configuration/llm-providers)

**Memobase**
- https://github.com/memodb-io/memobase, Apache-2.0, about 2.9k stars, 467 commits — [GitHub](https://github.com/memodb-io/memobase)
- Requires PostgreSQL + Redis (FastAPI server, dockerized). Integrates through the OpenAI SDK or OpenAI-compatible APIs — [GitHub](https://github.com/memodb-io/memobase)

**MemoryOS**
- https://github.com/BAI-LAB/MemoryOS. Ships an MCP server and a playground — [GitHub](https://github.com/BAI-LAB/MemoryOS)

**A-MEM**
- https://github.com/WujiangXu/A-mem, the official NeurIPS 2025 code — [GitHub](https://github.com/WujiangXu/A-mem)

**LightMem**
- https://github.com/zjunlp/LightMem. Advertises support for both cloud APIs and local models — [GitHub](https://github.com/zjunlp/LightMem)

**EverMemOS**
- Renamed EverOS: https://github.com/EverMind-AI/EverMemOS, Apache-2.0, about 13.3k stars — [GitHub](https://github.com/EverMind-AI/EverMemOS)
- Local-first, with Markdown files as the source of truth, indexed with SQLite + LanceDB. It does not need MongoDB, Elasticsearch or Redis. The quickstart requires only an OpenRouter key — [GitHub](https://github.com/EverMind-AI/EverMemOS)

**Supermemory**
- https://github.com/supermemoryai/supermemory, "can be run fully locally" per the repo description — [GitHub](https://github.com/supermemoryai/supermemory)
- Its open-source MemoryBench harness compares Supermemory, Mem0, Zep and others — [Supermemory research](https://supermemory.ai/research/longmembench/)

**Mem0 benchmark harness**
- github.com/mem0ai/memory-benchmarks — [Mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)

### Inferences
- **Gemini/Vertex ease, from easiest:**
  1. MIRIX and Cognee: env-var configuration, and Vertex is documented.
  2. Graphiti: native Gemini clients for all three roles.
  3. Mem0: Google AI LLM, Google AI or Vertex AI embeddings; Vertex LLMs likely go through the LiteLLM provider.
  4. Letta: Gemini and Vertex providers.
  5. LangMem: any LangChain chat model, so `google_genai` or `google_vertexai` should work. This is not explicitly documented on the page fetched.
  6. Memobase and MemOS: Gemini would likely require an OpenAI-compatible endpoint, such as Gemini's OpenAI-compatible API or LiteLLM. Not verified.
- **Ingesting a custom daily-conversation dataset:** all of these ingest (user_id, messages, timestamp) streams.
  - Graphiti/Zep need a correct per-message reference time (`created_at` or the episode reference time) and correct speaker roles. Zep's rebuttal shows that getting these wrong badly degrades its scores.
  - Mem0 takes message pairs with user_id and metadata.
  - Letta requires running a server and driving an agent, which makes it slower and more expensive per ingested message because every turn is an agent step.
- **Cost of per-message LLM calls:** heavy for Mem0 (extraction plus update), Graphiti (entity/edge extraction plus deduplication and invalidation), MemOS and MIRIX (multiple agents). Memobase batches messages with a fixed 3 calls per flush. Budget for this on long daily logs.

### Gaps
- Exact Gemini configuration for Memobase, MemOS, MemoryOS, A-MEM and LightMem was not verified.
- Exact licenses and star counts for MemoryOS, A-MEM, LightMem, Hindsight, MemMachine and Supermemory were not fetched. Hindsight's arXiv listing is CC BY 4.0 (the paper, not the code).
- Latest release dates were not visible for Mem0, Graphiti, Letta, MemOS, MIRIX and Memobase. Only Cognee (v1.6.1, 24 Sep 2026) was confirmed.
- Whether Mem0 OSS still ships graph memory (Mem0^g with Neo4j/Memgraph) after the April 2026 algorithm change was not confirmed from the fetched README.
