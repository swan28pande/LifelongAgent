# Temporal, Preference-Evolving, Hierarchical and Agentic LLM Memory (2024-2026): Baselines Comparable to a SQL + Hierarchical-Summary + Agentic-Retriever Design

Scope note: the target design stores dated preferences/facts/events in SQL, builds weekly->monthly->yearly->lifetime summaries, and retrieves with a tool-calling agent. Items marked **[verified]** had their venue confirmed this session from ACL Anthology / ICLR proceedings / OpenReview / arXiv comments. Items marked **[venue unverified]** give the venue from secondary sources or prior knowledge and should be checked before citing. PDF links are arXiv/Anthology/proceedings PDFs. Licenses and star counts come from the GitHub API, queried 2026-09-28.

## Q1. Memory systems that explicitly model temporal validity, knowledge updates, and conflict resolution

### Takeaway
The closest temporal baselines are **Zep/Graphiti** (bi-temporal KG, arXiv-only, Apache-2.0), **TReMu** (timeline summaries with inferred dates plus code-executed date arithmetic, Findings of ACL 2025), **TiMem** (Temporal Memory Tree, Findings of ACL 2026), and **MemForest** (time-ordered tree index, VLDB submission). **Memory-R1** (ACL 2026) is the main method that learns ADD/UPDATE/DELETE/NOOP overwrite-vs-append decisions. LongMemEval's "knowledge update" and "temporal reasoning" categories are the de facto evaluation. Several 2026 papers (STALE, Supersede, MemConflict, TOKI) show that update and conflict handling is still largely unsolved.

### Cited Findings
- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (Rasmussen et al., arXiv 2501.13956, Jan 2025; **arXiv-only**, industry paper). Graphiti is a temporally aware KG engine that combines conversational and business data "while preserving historical relationships". DMR: 94.8% vs MemGPT 93.4%. LongMemEval: accuracy gains "up to 18.5%" with 90% lower response latency than baseline implementations. PDF: https://arxiv.org/pdf/2501.13956. — [arXiv abs](https://arxiv.org/abs/2501.13956)
- Graphiti code: github.com/getzep/graphiti, **Apache-2.0**, about 31k stars. — [GitHub API](https://github.com/getzep/graphiti)
- A 2026 third-party evaluation reports that Graphiti (bi-temporal edges, validity intervals) scored only **7.0% on conflict-resolution tasks** despite strong LongMemEval numbers. This is a useful counterpoint when comparing against Zep. — [search summary of 2026 conflict-memory literature, e.g. MemConflict](https://arxiv.org/pdf/2605.20926) (attribution to the specific paper is not verified; check before citing)
- **TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues** (Ge et al.), **Findings of ACL 2025 [verified]**. Uses "time-aware memorization through timeline summarization": each session is summarized into events with **inferred dates**, then the LLM writes Python code for temporal calculations. It also releases a multi-session temporal QA benchmark covering relative time and cross-session dependencies. GPT-4o improves from **29.83 (standard prompting) to 77.67 (TReMu)**. PDF: https://aclanthology.org/2025.findings-acl.972.pdf, arXiv https://arxiv.org/pdf/2502.01630. — [ACL Anthology](https://aclanthology.org/2025.findings-acl.972/)
- **TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents** (Li et al., arXiv 2601.02845, Jan 2026; **Findings of ACL 2026 per arXiv metadata**). Organizes conversation into a **Temporal Memory Tree (TMT)** that runs from raw observations up to abstracted persona representations, with temporal organization as the primary principle. It uses semantic-guided consolidation (no fine-tuning) and complexity-aware recall. **LoCoMo 75.30%** (with 52.20% memory-length reduction), **LongMemEval-S 76.88%**. Code: github.com/TiMEM-AI/timem; arXiv lists CC-BY-4.0 for the paper, and the GitHub API returns license "NOASSERTION". PDF: https://arxiv.org/pdf/2601.02845. — [arXiv abs](https://arxiv.org/abs/2601.02845)
- **MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing** (Chen et al., arXiv 2605.23986, May 2026, rev. Sep 2026; **VLDB submission, not yet accepted**). Treats memory as "a write-efficient temporal data-management problem". A MemTree-style **time-ordered tree index** uses localized dirty-path refreshes instead of global summary rewrites. LongMemEval-S **81.8% pass@1** (Qwen3-30B) with a 6.0x faster build than EverMemOS; LoCoMo (cat. 1-4) **84.09%**. No code link found. PDF: https://arxiv.org/pdf/2605.23986. — [arXiv abs](https://arxiv.org/abs/2605.23986)
- **SodaMem: Evidence-Grounded Temporal Graph Memory for LLM Agents** (arXiv 2608.08055, 2026). Found in search but not read in detail. PDF: https://arxiv.org/pdf/2608.08055. — [arXiv PDF](https://arxiv.org/pdf/2608.08055)
- **Memory-R1** (arXiv 2508.19828), **ACL 2026 Long [verified via Anthology entry 2026.acl-long.583]**. Uses an RL-trained Memory Manager that picks among **ADD, UPDATE, DELETE, NOOP** (explicit overwrite vs append), plus an Answer Agent that pre-selects and reasons over retrieved entries. Trained with PPO/GRPO from outcome rewards. PDF: https://arxiv.org/pdf/2508.19828. — [ACL Anthology](https://aclanthology.org/2026.acl-long.583/); [arXiv](https://arxiv.org/abs/2508.19828)
- **EverMemOS** (arXiv 2601.02163, Jan 2026; arXiv-only as far as found). Builds MemCells (episodic traces, atomic facts, and **time-bounded "Foresight" signals**), consolidates them into MemScenes, updates the user profile, and uses **MemScene-guided agentic retrieval**. Reported LoCoMo 86.76% (GPT-4o-mini) / 93.05% (GPT-4.1-mini); LongMemEval 83.00% vs MemOS 77.80%. Code: github.com/EverMind-AI/EverMemOS, Apache-2.0, about 13k stars. PDF: https://arxiv.org/pdf/2601.02163. — [HF paper page](https://huggingface.co/papers/2601.02163); [arXiv](https://arxiv.org/abs/2601.02163)
- **LongMemEval** (Wu et al., **ICLR 2025**) has 500 questions in six categories, including **knowledge update** (a fact stated in one session changes later) and **temporal reasoning**, plus abstention. Code: xiaowu0162/LongMemEval, MIT. PDF: https://arxiv.org/pdf/2410.10813. — [supermemory summary](https://supermemory.ai/research/longmembench/); [mem0 benchmark guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- **LoCoMo** (Maharana et al., ACL 2024; standard older reference) has 1,540 QA items across single-hop, multi-hop, open-domain and **temporal** categories. Code: snap-research/locomo (license NOASSERTION). PDF: https://arxiv.org/pdf/2402.17753. — [mem0 benchmark guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- 2026 benchmarks and methods for memory invalidation (arXiv-only, found but not read in depth):
  - **STALE**: "Can LLM Agents Know When Their Memories Are No Longer Valid?", arXiv 2605.06527.
  - **Supersede**: memory-update gap, arXiv 2606.27472.
  - **TOKI**: bitemporal operator algebra for contradiction resolution, arXiv 2606.06240.
  - **MemConflict**: arXiv 2605.20926.
  - Deterministic freshness recipe: arXiv 2606.01435.
  - MemoryAgentBench (Hu et al., 2025, arXiv 2507.05257) and MemBench list knowledge updating and conflict resolution as core competencies.
  - Sources: [STALE](https://arxiv.org/html/2605.06527v1); [Supersede](https://arxiv.org/html/2606.27472v1); [TOKI](https://arxiv.org/pdf/2606.06240); [MemConflict](https://arxiv.org/pdf/2605.20926); [Freshness](https://arxiv.org/html/2606.01435v1); [MemoryAgentBench](https://arxiv.org/pdf/2507.05257)

### Inferences
- TReMu is the most direct comparison for storing dated events and computing over dates: its code-executed temporal arithmetic is conceptually close to SQL date queries. A SQL store with explicit dates should be framed as a structured generalization of TReMu's timeline memory.
- Zep/Graphiti is the canonical "temporal validity" baseline (valid_from/valid_to on edges). Reviewers will expect a comparison against it on LongMemEval knowledge-update and temporal-reasoning questions.
- TiMem and MemForest are the nearest competitors to the weekly->monthly->yearly hierarchy, since both organize memory as time-ordered trees. The novelty argument has to set the new approach apart from them, for example with SQL-backed facts/preferences plus calendar-aligned summary levels plus an agentic retriever.

### Gaps
- Whether the Graphiti 7.0% conflict-resolution figure comes from MemConflict or another 2026 paper was not confirmed.
- TReMu and Memory-R1 code repositories and licenses were not located this session.
- TiMem's exact tree levels (e.g. whether they are calendar-aligned day/week/month) were not visible in the abstract. Read the PDF to confirm.

## Q2. Personalization and preference tracking over time (methods and benchmarks)

### Takeaway
The key benchmarks are **PrefEval** (ICLR 2025 oral), **PersonaMem** (COLM 2025) and **HorizonBench** (arXiv 2026, 6-month histories with preference shifts), plus LongMemEval's preference and knowledge-update categories. All three show frontier LLMs near or below 50% on evolving-preference tracking. Methods that explicitly target personalization include MemoryOS (EMNLP 2025), SeCom (ICLR 2025), TiMem's persona layer, and EverMemOS's profile updating.

### Cited Findings
- **PrefEval: "Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs"** (Zhao et al., Amazon/UMN), **ICLR 2025 Oral [verified]**. It has 3,000 curated preference/query pairs over 20 topics, with explicit and implicit preferences and multi-session contexts up to 100k tokens. In zero-shot settings, preference following falls **below 10% at just 10 turns (~3k tokens)** for most models, and it still degrades with prompting and RAG. Fine-tuning helps. Code: amazon-science/PrefEval (license NOASSERTION); site prefeval.github.io. PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/28a46044775d97a4efcbcf14e7f13209-Paper-Conference.pdf (arXiv 2502.09597). — [ICLR virtual](https://iclr.cc/virtual/2025/oral/31848); [OpenReview](https://openreview.net/forum?id=QWunLKbBGF)
- **PersonaMem: "Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale"** (Jiang et al., arXiv 2504.14225), **COLM 2025** (per repo title). A NeurIPS 2025 virtual page also lists it, possibly as a satellite/locale presentation, so the venue should be confirmed. It has 180+ simulated users, histories of up to 60 sessions, and 15 tasks. The questions test whether models track **how preferences evolve over time**. GPT-4.1, o4-mini, GPT-4.5, o1 and Gemini-2.0 reach only **~50% overall accuracy**. Code: bowen-upenn/PersonaMem, MIT. PDF: https://arxiv.org/pdf/2504.14225. — [GitHub](https://github.com/bowen-upenn/PersonaMem); [arXiv](https://arxiv.org/abs/2504.14225); [NeurIPS page](https://neurips.cc/virtual/2025/loc/san-diego/122431)
- **HorizonBench: Long-Horizon Personalization with Evolving Preferences** (Li, Paranjape, ..., Tsvetkov, Celikyilmaz; arXiv 2604.17283, Apr 2026; **arXiv-only**). It has 4,245 items from 360 simulated users with **6-month histories** (~4,300 turns, ~163K tokens), generated from mental-state graphs that give ground truth for preference shifts. The best model scores **52.8%**. "Over a third of the time" models pick the originally stated value instead of the updated one. It evaluates 25 models, including memory-augmented systems. PDF: https://arxiv.org/pdf/2604.17283. — [arXiv](https://arxiv.org/abs/2604.17283)
- **SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents**, **ICLR 2025** (OpenReview xKDZAW0He3; arXiv 2502.05589). Uses segment-level memory units plus compression. PDF: https://openreview.net/pdf?id=xKDZAW0He3. — [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- Related 2026 preference work (arXiv-only, not read in detail):
  - Latent Preference Modeling for Multi-Session Personalized Tool Calling, arXiv 2604.17886.
  - Hierarchical Strategy Co-Evolution for Agent Memory, arXiv 2608.25329.
  - Sources: [2604.17886](https://arxiv.org/pdf/2604.17886); [2608.25329](https://arxiv.org/pdf/2608.25329)

### Inferences
- A SQL preference table with dates and supersession maps directly onto HorizonBench's main failure mode: returning the originally stated value instead of the update. HorizonBench and PersonaMem are therefore the most persuasive evaluation targets for the "evolving preferences" claim. PrefEval tests preference adherence rather than change over time.

### Gaps
- No verified method paper was found that reports SOTA specifically on HorizonBench. Its baseline list (which memory systems) was not extracted.
- PersonaMem's venue shows as both COLM 2025 and a NeurIPS 2025 page. Confirm via the COLM proceedings/OpenReview.

## Q3. Hierarchical or multi-granularity summaries, and structured/SQL/table memory

### Takeaway
Hierarchical baselines to cite are:
- **MemoryOS** (EMNLP 2025 oral): short-, mid- and long-term tiers.
- **TiMem** (Findings of ACL 2026): temporal tree.
- **MemForest** (2026): temporal tree index.
- **MemTree** (arXiv 2024): semantic tree.
- **RAPTOR** (ICLR 2024): recursive clustering and summarization.
- **RSum** (Neurocomputing 2025): recursive summarization.
- **MemoryBank** (AAAI 2024): Ebbinghaus-style forgetting.

The canonical SQL memory is **ChatDB** (arXiv 2023, no license). No 2025-2026 peer-reviewed conversational-memory method was found that uses a relational SQL store with calendar-aligned hierarchical summaries, which looks like the novelty gap.

### Cited Findings
- **MemoryOS: "Memory OS of AI Agent"** (Kang et al., BAI-LAB), **EMNLP 2025 main, oral [verified]**. It has three storage tiers (short-term, mid-term, long-term personal memory) and four modules (storage, updating, retrieval, generation). Short-to-mid updates use dialogue-chain FIFO; mid-to-long updates use segmented pages. LoCoMo gains average **+48.36% F1 and +46.18% BLEU-1** over baselines on GPT-4o-mini. Code: BAI-LAB/MemoryOS, Apache-2.0. PDF: https://aclanthology.org/2025.emnlp-main.1318.pdf. — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1318/); [GitHub](https://github.com/BAI-LAB/MemoryOS)
- **MemTree: "From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs"** (Rezazadeh et al., Accenture; arXiv 2410.14052, Oct 2024). A dynamic tree: each node holds aggregated text, an embedding and an abstraction level, and new information is inserted by embedding similarity. An OpenReview forum exists (moXtEmCleY), but acceptance was not confirmed, so treat it as **arXiv-only**. Only an unofficial implementation exists (Feesuu/MemoryTree). PDF: https://arxiv.org/pdf/2410.14052. — [arXiv](https://arxiv.org/abs/2410.14052); [OpenReview](https://openreview.net/forum?id=moXtEmCleY)
- **RSum: "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models"** (Q. Wang et al.; arXiv 2308.15022), **Neurocomputing 2025** (doi 10.1016/j.neucom.2025.130193) **[verified]**. The LLM repeatedly produces a new memory from the previous memory plus new context, which complements long-context and retrieval LLMs. PDF: https://arxiv.org/pdf/2308.15022. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225008653); [arXiv](https://arxiv.org/abs/2308.15022)
- **TiMem** and **MemForest**: temporal trees (see Q1). **EverMemOS**: MemCell->MemScene consolidation (see Q1).
- **LightMem** (zjunlp), **ICLR 2026** (per repo title). Lightweight memory-augmented generation with a sensory -> short-term -> long-term pipeline and offline "sleep-time" consolidation (architecture from prior knowledge, not verified this session). Code MIT. PDF: https://arxiv.org/pdf/2510.18866. — [GitHub](https://github.com/zjunlp/LightMem)
- Older standard references, 2023-2024 **[venue unverified this session; from prior knowledge]**:
  - **RAPTOR**: recursive abstractive tree over clustered chunks. ICLR 2024; code parthsarthi03/raptor, MIT; PDF https://arxiv.org/pdf/2401.18059.
  - **MemoryBank**: memory with Ebbinghaus forgetting curve plus user-portrait summaries. AAAI 2024; code zhongwanjun/MemoryBank-SiliconFriend, MIT; PDF https://arxiv.org/pdf/2305.10250.
  - Licenses verified via the GitHub API. — [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory** (Hu et al., arXiv 2306.03901, 2023; **arXiv-only**, standard reference). The LLM generates SQL instructions to read and write a relational DB as symbolic memory ("chain-of-memory"). Code: huchenxucs/ChatDB, **no license**. PDF: https://arxiv.org/pdf/2306.03901. — [search summary](https://arxiv.org/pdf/2602.11243); [GitHub API](https://github.com/huchenxucs/ChatDB)
- Other structured-store evidence:
  - Letta persists memory in Postgres/SQLite (core blocks in the prompt, archival memory in the DB).
  - "MinnsDB", a commercial memory database, stores facts as graph edges with valid_from/valid_until. This is an industry claim with no paper.
  - "Evaluating Memory Structure in LLM Agents" (arXiv 2602.11243) compares memory structures.
  - Sources: [DEV Community list](https://dev.to/jonathanfarrow/the-10-best-ai-memory-layers-for-agents-in-2026-448e) (low-quality source); [2602.11243](https://arxiv.org/pdf/2602.11243)

### Inferences
- The strongest "closest prior" for SQL memory is ChatDB, which predates the LLM-memory benchmarks and is not evaluated on LoCoMo or LongMemEval. That leaves room to show SQL memory with dated rows on modern benchmarks.
- MemoryOS's tiers are recency/heat based, not calendar based. TiMem and MemForest are time based. A calendar-aligned weekly/monthly/yearly/lifetime hierarchy should be contrasted explicitly with all three.

### Gaps
- RAPTOR, MemoryBank and LoCoMo venues were not re-verified this session (they are well known as ICLR 2024, AAAI 2024 and ACL 2024).
- No 2025-2026 peer-reviewed paper was found that combines a relational SQL store with time-hierarchical summaries. Absence of evidence comes from limited search, not an exhaustive review.

## Q4. Agentic, tool-calling retrieval over memory

### Takeaway
The main agentic-retrieval baselines are:
- **MemGPT/Letta** (arXiv 2023, Apache-2.0): memory tools called by the LLM.
- **MIRIX** (arXiv 2025, Apache-2.0): six memory types with a multi-agent router.
- **Memory-R1** (ACL 2026): RL-trained manager and answer agents.
- **EverMemOS**: agentic retrieval guided by MemScenes.
- **Mem-α** (RL-learned memory construction).

Mem0 (ECAI 2025 per Mem0's own blog) and A-MEM are the most common non-agentic comparison baselines.

### Cited Findings
- **MemGPT: Towards LLMs as Operating Systems** (Packer et al., arXiv 2310.08560, 2023; **arXiv-only**, standard reference). OS-style paging between in-context and external stores via LLM-issued function calls. Now the **Letta** framework: letta-ai/letta, Apache-2.0, about 25k stars. PDF: https://arxiv.org/pdf/2310.08560. — [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List); [search summary](https://arxiv.org/html/2603.07670v1)
- **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (arXiv 2507.07957, Jul 2025; **no peer-reviewed venue found**). Six memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) managed by specialized agents. Used as a baseline in ICLR 2026 papers. Code: Mirix-AI/MIRIX, Apache-2.0. PDF: https://arxiv.org/pdf/2507.07957. — [search summary](https://arxiv.org/html/2603.07670v1); [ICLR 2026 paper using MIRIX as baseline](https://proceedings.iclr.cc/paper_files/paper/2026/file/fd1eff9dd295df50a41f2521942fa31d-Paper-Conference.pdf)
- **Memory-R1**, ACL 2026 (see Q1). The Answer Agent learns to filter retrieved memories, and the Manager learns write operations.
- **Mem-α: Learning Memory Construction via Reinforcement Learning** (arXiv 2509.25911). An OpenReview PDF exists; the venue (likely ICLR 2026) was not confirmed. PDF: https://arxiv.org/pdf/2509.25911. — [OpenReview PDF](https://openreview.net/pdf/84b195754f5a425454f70a545ce1e22ee38834db.pdf)
- **MemR^3: Memory Retrieval via Reflective Reasoning for LLM Agents** (arXiv 2512.20237). Iterative, reasoning-driven retrieval (not read in detail). PDF: https://arxiv.org/pdf/2512.20237. — [arXiv PDF](https://arxiv.org/pdf/2512.20237)
- **Mem0** (arXiv 2504.19413). Extract-then-update pipeline (ADD/UPDATE/DELETE via LLM) with a graph variant. Mem0's own blog says it was published at **ECAI 2025** and gives a 10-method comparison on LoCoMo; this is a self-reported claim. Code Apache-2.0, about 66k stars. PDF: https://arxiv.org/pdf/2504.19413. — [mem0 blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- **A-MEM: Agentic Memory for LLM Agents** (arXiv 2502.12110). Zettelkasten-style notes with LLM-generated attributes and links. Reported at NeurIPS 2025 (**venue unverified this session**). Code agiresearch/A-mem, MIT. PDF: https://arxiv.org/pdf/2502.12110. — [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- **Nemori** (arXiv 2508.03341), a self-organizing episodic memory, is also listed; venue unknown. PDF: https://arxiv.org/pdf/2508.03341. — [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- Useful 2026 surveys:
  - "Memory in the Age of AI Agents: A Survey" (paper list at Shichun-Liu/Agent-Memory-Paper-List).
  - "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers" (arXiv 2603.07670).
  - "Anatomy of Agentic Memory" (arXiv 2602.19320).
  - Sources: [2603.07670](https://arxiv.org/html/2603.07670v1); [2602.19320](https://arxiv.org/pdf/2602.19320)

### Inferences
- MemGPT/Letta is the obvious "tool-calling retriever" baseline. Memory-R1 is the peer-reviewed 2026 counterpart with learned policies. A prompted (non-RL) tool-calling agent over SQL should be compared against both, and the paper should argue why structured SQL tools beat free-text archival search.
- A minimal comparison set for the target method:
  - Zep/Graphiti (temporal validity)
  - TReMu (dated timelines)
  - TiMem and/or MemForest (temporal hierarchy)
  - MemoryOS (hierarchical tiers, personalization)
  - MemGPT/Letta and Memory-R1 (agentic retrieval and updates)
  - Mem0 and A-MEM (standard baselines)
  - ChatDB (SQL memory prior)
  - Evaluate on LongMemEval (knowledge-update and temporal categories), LoCoMo, PersonaMem and HorizonBench.

### Gaps
- MIRIX's benchmark numbers and any venue acceptance were not retrieved.
- Mem0's ECAI 2025 and A-MEM's NeurIPS 2025 claims need checking against the proceedings.
- Memory-R1 code/license and headline LoCoMo numbers were not extracted.
