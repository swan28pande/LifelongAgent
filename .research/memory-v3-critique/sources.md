# Sources

## Repository evidence

- [Current v3 agent and read path](../../memory_v3/agent.py), [ingestion](../../memory_v3/ingest.py), [prompts](../../memory_v3/prompts.py), [retrieval tools](../../memory_v3/tools.py), [SQLite store](../../memory_v2/store.py), and [hierarchical summarizer](../../memory_v2/summarizer.py).
- [Synthetic dataset](../../datasets/eval/conversations.json), [saved synthetic run](../../results/synthetic/v3/README.md), and [question-level outputs](../../results/synthetic/v3/qa_results.json).
- [Saved LoCoMo comparison](../../results/locomo/README.md) and [evaluation protocol](../../benchmarks/locomo/README.md).
- [Previous related-work review](../memory-v3-prior-art/report.md). Rechecked claims used here against primary papers.

## Primary research sources

- [LongMemEval, ICLR 2025](https://arxiv.org/pdf/2410.10813): long-history QA, oracle/full-context comparisons, and fact-augmented retrieval. Figures 3a–b; Tables 3–4.
- [Mem0, 2025](https://arxiv.org/pdf/2504.19413): LoCoMo full-context, RAG, and memory comparisons; graph ablation and latency. Tables 1–2.
- [APEX-MEM, ACL 2026](https://aclanthology.org/2026.acl-long.749.pdf): temporally grounded graph memory with source evidence and multi-tool reader; LoCoMo and LongMemEval results. §§3–4, Tables 1, 3–4, 9–10.
- [TiMem, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1091.pdf): adaptive temporal hierarchy and ablations. Tables 2–6.
- [Retrieval Quality at Context Limit, 2025](https://research.google/pubs/retrieval-quality-at-context-limit/): newer-model counter-evidence for simple long-context factoid retrieval.
- [RHELM, 2026 preprint](https://arxiv.org/pdf/2605.31086): evolving, heterogeneous histories; both full context and retrieval remain weak on its harder benchmark. Table 4. Used only as a scope check, not a head-to-head with v3.
- [Beyond the Context Window, March 2026 preprint](https://arxiv.org/pdf/2603.04814): GPT-5-mini full-history versus a flat Mem0 pipeline on LoCoMo and LongMemEval, with an LLM judge and modeled repeated-query cost. Table 3; §§3–4. Not peer-reviewed and not a comparison with v3.
