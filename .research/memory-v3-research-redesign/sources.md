# Sources for the research redesign

Date checked: 2026-09-29. Primary papers, official proceedings, and repository source code only. This is a bounded scan, not an exhaustive bibliography.

## Local primary sources

- [Synthetic dataset generator](../../scripts/generate_eval_dataset.py): schedule rules, conversation prompts, validation, QA sampling and gold answers.
- [Saved synthetic conversations](../../datasets/eval/conversations.json) and [QA](../../datasets/eval/qa_pairs.json): generated material actually evaluated.
- [Synthetic v3 evaluator](../../benchmarks/synthetic/run_v3.py): ingestion timing and scorers.
- [Legacy synthetic evaluator](../../evaluation/evaluate_synthetic.py): comparator's default ingestion window.
- [Saved v3 synthetic responses](../../results/synthetic/v3/qa_results.json) and [LoCoMo report](../../results/locomo/README.md): observed local results and protocol limits.
- [Store](../../memory_v2/store.py), [v3 agent](../../memory_v3/agent.py), and [prior architecture investigation](../memory-v3-architecture/report.md): implementation context.
- [Prior-art investigation](../memory-v3-prior-art/report.md) and [critique](../memory-v3-critique/report.md): earlier bounded scans, not substitutes for the papers below.

## Papers and official resources

| Research area | Primary source | Relevance |
|---|---|---|
| Evolving preferences | [HorizonBench (2026)](https://arxiv.org/html/2604.17283) | Six-month histories, implicit state changes, hard negatives, full-context models. |
| Evolving preferences | [PERMA (2026)](https://arxiv.org/html/2603.23231) | Event-driven and interim preference probes, interactive evaluation. |
| Implicit personalization | [PersonaMem-v2 (2025)](https://arxiv.org/html/2512.06688) | Synthetic implicit preferences and updates. |
| Preference following | [PrefEval (ICLR 2025)](https://openreview.net/pdf?id=QWunLKbBGF) | Whether an assistant follows expressed preferences in responses. |
| Memory lifecycle | [MemOps (2026)](https://arxiv.org/html/2607.12893) | Remember, update, forget, reflect operation traces and evidence. |
| Obsolete information | [Memora (2026)](https://arxiv.org/html/2604.20006) | Penalties for acting on superseded memory. |
| Ambiguous state changes | [CAPTURE (2026 preprint)](https://arxiv.org/html/2609.02265) | Drift, temporary context, malicious updates, clarification. |
| Memory-to-action | [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) | Memory-dependent tool calls and arguments. |
| Memory-to-action | [PersonalAlign / AndroidIntent (ACL 2026)](https://aclanthology.org/2026.acl-long.1669/) | Vague instructions, personal preferences, routines, proactive suggestions. |
| Natural conversation | [RealTalk (2025)](https://arxiv.org/html/2502.13270) | Human conversation rather than only generated histories. |
| Very long agent histories | [LongMemEval-V2 (2026)](https://arxiv.org/html/2605.12493) | Environment state, workflows, gotchas across up to 115M tokens. |
| Longitudinal behavior | [MemoryCD (2026)](https://arxiv.org/html/2603.25973) | Real-user behavioral histories for personalization. |
| Preference forecasting | [LATTE (2026)](https://arxiv.org/html/2605.26612) | Preference trajectory forecasting for personalized generation. |
| Feedback and decisions | [Li et al., contextual bandits (2010)](https://arxiv.org/abs/1003.0146) | Older personalization line that optimizes decisions from sequential user feedback. |

For the closest architecture papers, see the [prior-art source list](../memory-v3-prior-art/sources.md), especially APEX-MEM, Chronos, TiMem, Graphiti/Zep, and Hindsight.
