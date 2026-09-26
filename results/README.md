# Results

Organised as `results/<benchmark>/<system>/`. Every run folder has its own README saying
which command produced it, its scores, and what each file contains.

```
results/
├── locomo/                 LoCoMo benchmark: benchmarks/locomo/
│   ├── v3_agentic/         memory_v3, tool-calling agent
│   ├── v3_onepass/         memory_v3 store, one retrieval + one LLM call
│   └── mem0/               mem0 baseline, one retrieval + one LLM call
├── synthetic/              synthetic preference benchmark: benchmarks/synthetic/
│   └── v3/                 memory_v3
└── legacy/                 earlier systems and superseded runs (memory_v2, baselines)
```

## Current numbers

**LoCoMo**, conv-26, 199 questions, paper token F1 (details and caveats in
[locomo/README.md](locomo/README.md)):

| System | Overall | Excluding adversarial | Calls/question |
|---|---|---|---|
| v3_agentic | 0.460 | **0.557** | ~8 |
| v3_onepass | **0.608** | 0.546 | 1 |
| mem0 | 0.557 | 0.452 | 1 |

**Synthetic**, 67 questions ([synthetic/README.md](synthetic/README.md)): memory_v3 token
F1 0.143, LLM judge 0.478.

## Conventions

- Each LoCoMo run has `locomo_results.json` (settings, summary and every record),
  `<sample_id>_trace.jsonl` (one line per answer, written as it goes) and, where the run
  ingests, `stores/<sample_id>/`.
- A new run of an existing system overwrites its folder. Pass `--run <name>` to keep
  both, and add a README to the new folder.
