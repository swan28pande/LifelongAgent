# Legacy results

Outputs from earlier systems and superseded runs, kept for reference. Nothing here
is produced by the current `benchmarks/` runners.

| Folder | What it is | Produced by |
|---|---|---|
| `locomo_v3_partial/` | First memory_v3 LoCoMo run on conv-26, before the `semantic_search_summaries` tool existed. Stalled after 24 questions, so it has a trace only and no results file. | old `scripts/run_locomo_v3.py` |
| `locomo_v2/` | memory_v2 LoCoMo results (older record format: one nested object per system). | `evaluation/evaluate_locomo.py` |
| `locomo_debug/` | Store from a LoCoMo debugging run. | ad hoc |
| `complete_comparison/` | Synthetic-dataset comparison of memory_v2, NaiveRAG, RSum, MemoryBank and MemGPT on Gemini Flash and Pro, with each system's summaries and working context. | `evaluation/evaluate_synthetic.py` |
| `memory_v2_run/` | memory_v2 store and lifetime summary from the full pipeline run. | `scripts/run_full_gpt5_pipeline.py` |
| `gemini_extraction_run/`, `gemini_flash_run/` | memory_v2 runs on the synthetic dataset with different Gemini models: extracted memories, summaries, QA results, store. | `scripts/run_gemini_extraction.py`, `scripts/run_gemini_flash.py` |
| `verify_run/` | Scratch store from checking extraction against the dataset's ground-truth patterns. | `scripts/verify_extraction.py` |
| `logs/` | Console logs from earlier v3 runs. | — |
| `loose_outputs/` | Single-file outputs that used to sit directly in `results/` (extraction dumps, summaries, the RSum memory, an HTML presentation, early synthetic results). | various |

The `evaluation/` scripts for MSC and probing write to `legacy/msc/` and
`legacy/probing/` when run.
