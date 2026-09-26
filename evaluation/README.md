# evaluation: older memory_v2 evaluators

These evaluate **memory_v2** against the baselines. Current systems are evaluated from
[benchmarks/](../benchmarks/) instead. Outputs go to `results/legacy/`.

| File | What it evaluates | Output |
|---|---|---|
| [evaluate_locomo.py](evaluate_locomo.py) | memory_v2 on LoCoMo with the paper's scoring, compared with the baselines. | `results/legacy/locomo_v2/locomo_results.json` |
| [evaluate_msc.py](evaluate_msc.py) | memory_v2 on MSC session 5, compared with the ChatGPT context-only and RSum predictions in `baselines/Rsum/results/`. | `results/legacy/msc/` |
| [evaluate_probing.py](evaluate_probing.py) | Probing on MSC personas: persona facts turned into factual QA by an LLM, then answered from memory. | `results/legacy/probing/` |
| [evaluate_synthetic.py](evaluate_synthetic.py) | memory_v2, NaiveRAG, RSum, MemoryBank and MemGPT on the synthetic dataset (30 days), Gemini Flash-Lite and Pro, F1 plus an LLM judge. | `results/legacy/complete_comparison/` |

Run from the project root, e.g. `venv/bin/python evaluation/evaluate_locomo.py`.
