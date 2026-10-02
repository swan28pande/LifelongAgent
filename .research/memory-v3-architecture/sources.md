# Sources for the memory v3 walkthrough

## Current local implementation

- [memory_v3/agent.py](../../memory_v3/agent.py): construction, batch and live entry points, reader invocation.
- [memory_v3/ingest.py](../../memory_v3/ingest.py): structured extraction, normalization, deduplication, writing, raw indexing.
- [memory_v3/tools.py](../../memory_v3/tools.py): six read-only tools.
- [memory_v3/prompts.py](../../memory_v3/prompts.py): memory type definitions and reader instructions.
- [memory_v2/store.py](../../memory_v2/store.py): SQLite schema and FAISS persistence.
- [memory_v2/summarizer.py](../../memory_v2/summarizer.py): period hierarchy and incremental updates.
- [tests/test_memory_v3.py](../../tests/test_memory_v3.py): offline plumbing tests and their stated limits.

## Dataset, runners, and saved artifacts

- [Synthetic generator](../../scripts/generate_eval_dataset.py), [conversations](../../datasets/eval/conversations.json), [questions](../../datasets/eval/qa_pairs.json), [runner](../../benchmarks/synthetic/run_v3.py), [saved results](../../results/synthetic/v3/README.md).
- [LoCoMo protocol](../../benchmarks/locomo/README.md), [agentic runner](../../benchmarks/locomo/run_v3_agentic.py), [one-pass runner](../../benchmarks/locomo/run_v3_onepass.py), [mem0 runner](../../benchmarks/locomo/run_mem0.py), [local scorer](../../benchmarks/locomo/common.py), [saved results](../../results/locomo/README.md).
- [Bundled LoCoMo evaluator](../../baselines/locomo/task_eval/evaluation.py): comparator for the repository's paper-exact scoring claim.
- [Legacy synthetic evaluator](../../evaluation/evaluate_synthetic.py): historical baseline protocol.

## Primary external sources

- [APEX-MEM, ACL 2026](https://aclanthology.org/2026.acl-long.749.pdf): append-only temporal graph, source evidence, multi-tool retrieval.
- [Chronos, arXiv 2026](https://arxiv.org/html/2603.16862): structured event calendar, raw turn calendar, iterative reader.
- [TiMem, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1091.pdf): temporal memory hierarchy and complexity-aware recall.
- [Original LoCoMo paper, ACL 2024](https://aclanthology.org/2024.acl-long.747/): benchmark definition.

The existing [prior-art investigation](../memory-v3-prior-art/report.md) offers broader positioning. This report verifies the three closest architectural comparisons from their primary papers and makes no cross-protocol numeric ranking.
