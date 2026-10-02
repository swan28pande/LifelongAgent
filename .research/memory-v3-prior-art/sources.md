# Sources

Cutoff: 2026-09-26. These are primary papers, official repositories, or official product documents. Vendor performance statements are labeled as such in the evidence and report.

## System papers and implementations

| ID | Source | Locations used |
|---|---|---|
| S01 | Banerjee et al., [APEX-MEM](https://aclanthology.org/2026.acl-long.749/) (ACL 2026); [PDF](https://aclanthology.org/2026.acl-long.749.pdf) | Abstract; §§3–4, 6.1–6.3; Tables 1, 3, 4; Appendix F and limitations |
| S02 | [Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval](https://arxiv.org/html/2603.16862) (2026 preprint) | §§3.1–3.5, 4; Tables 1–3 |
| S03 | [REMem: Reasoning with Episodic Memory in Language Agent](https://arxiv.org/html/2602.13530) (ICLR 2026); [official repository](https://github.com/intuit-ai-research/REMem) | §§3.1–3.2, 4; Table 2 |
| S04 | Ong et al., [Towards Lifelong Dialogue Agents via Timeline-based Memory Management (THEANINE)](https://aclanthology.org/2025.naacl-long.435/) (NAACL 2025); [full text](https://arxiv.org/html/2406.10996) | §§2–4 |
| S05 | Rasmussen et al., [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/html/2501.13956) (2025 preprint); [Graphiti official repository](https://github.com/getzep/graphiti) | §§2.2–2.3, 4.2–4.3; Table 2; README |
| S06 | Li et al., [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](https://aclanthology.org/2026.findings-acl.1091/) (Findings ACL 2026); [PDF](https://aclanthology.org/2026.findings-acl.1091.pdf) | §§3.1–3.3, 4.1, 6; Table 1 |
| S07 | [HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents](https://arxiv.org/abs/2601.06377) (2026 preprint); [PDF](https://arxiv.org/pdf/2601.06377) | §§2.2–2.4, 3; Table 1 |
| S08 | [H-Mem: Hybrid Multi-Dimensional Memory Management for Long-Context Conversational Agents](https://aclanthology.org/2026.eacl-long.363/) (EACL 2026); [PDF](https://aclanthology.org/2026.eacl-long.363.pdf) | §§3.1–3.2, 4.4; Table 1 |
| S09 | [HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](https://arxiv.org/abs/2604.18349) (2026 preprint); [PDF](https://arxiv.org/pdf/2604.18349) | §§3.1–3.3, 4.1; Tables 1–2 |
| S10 | [HINDSIGHT: Structured Agent Memory that Retains, Recalls, and Reflects](https://aclanthology.org/2026.acl-demo.27/) (ACL 2026 demo); [technical paper](https://arxiv.org/html/2512.12818); [official repository](https://github.com/vectorize-io/hindsight) | Technical paper §§3–4, 7; Tables 3–4; repository documentation |
| S11 | Chhikara et al., [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/html/2504.19413) (2025 preprint; ECAI 2025) | §§2–3; Tables 1–2 |
| S12 | Mem0, [current research page](https://mem0.ai/research), [May 2026 temporal update](https://mem0.ai/blog/the-token-efficient-memory-algorithm-now-has-temporal-reasoning), [official benchmark harness](https://github.com/mem0ai/memory-benchmarks) | Current managed-platform architecture and vendor-reported results; not interchangeable with S11 or the repository's local OSS baseline |
| S13 | [What Deserves Memory: How Agents Use Surprise to Build Long-Term Episodic Memory (NEMORI)](https://arxiv.org/abs/2508.03341) (first posted 2025, revised 2026) | §§3.2–3.4, 4.1; Table 2 |
| S14 | [PGMem: Tightly Coupled Persona-Memory Graph for Lifelong Personalized Agents](https://arxiv.org/html/2608.01708) (2026 preprint) | §§4–5; preference state/trait transitions; adjacent on preference evolution |
| S15 | Wang and Chen, [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/html/2507.07957) (2025 preprint); [official repository](https://github.com/Mirix-AI/MIRIX) | §§3.1–3.3, 4.1; six memory components, active retrieval, LoCoMo protocol |
| S16 | LangChain, [LangMem concepts](https://github.com/langchain-ai/langmem/blob/main/docs/docs/concepts/conceptual_guide.md), [official repository](https://github.com/langchain-ai/langmem) | Memory types, background consolidation, update/delete primitives and retrieval options |

## Benchmark sources

| ID | Source | Relevance |
|---|---|---|
| B01 | Maharana et al., [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](https://aclanthology.org/2024.acl-long.747/) (ACL 2024); [official data and code](https://github.com/snap-research/locomo) | Multi-session QA categories and original scoring; public `locomo10.json` split must be named explicitly |
| B02 | Wu et al., [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://openreview.net/pdf?id=pZiyCaVuti) (ICLR 2025); [official repository](https://github.com/xiaowu0162/LongMemEval) | 500-item S/M/oracle sets; preference, update, temporal, multi-session and abstention tasks |
| B03 | Jiang et al., [Know Me, Respond to Me / PERSONAMEM](https://arxiv.org/abs/2504.14225) (COLM 2025) | Up to 60 sessions and evolving user profiles in personalization decisions |
| B04 | Liu et al., [PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments](https://arxiv.org/html/2603.23231) (2026 preprint) | Event-driven preference evolution, temporal checkpoints and interactive tasks; §§4.1–4.3 |
| B05 | Xie et al., [DynamicMem: A Long-Horizon Memory Benchmark in Real-World Settings](https://arxiv.org/html/2606.22877) (2026 preprint) | 15-month multi-app evolution of attributes, habits and preferences; §§3–5 |
| B06 | Jiayang et al., [AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](https://arxiv.org/abs/2603.01966) (ICLR 2026) | On-policy simulated-user conversations with evolving state and write/read/utilization diagnostics |

## Local primary evidence

| ID | Source | Relevance |
|---|---|---|
| L01 | `memory_v3/agent.py`, `ingest.py`, `tools.py`, `prompts.py` | Current implementation |
| L02 | `memory_v2/store.py`, `summarizer.py` | Imported storage and summary hierarchy |
| L03 | `benchmarks/locomo/README.md`, `results/locomo/README.md` | Current public benchmark protocol and results |
| L04 | `benchmarks/synthetic/README.md`, `results/synthetic/v3/README.md`, `results/synthetic/v3/qa_results.json` | Current synthetic preference results |
