# Synthetic: memory_v3

**Produced by:** `benchmarks/synthetic/run_v3.py` (the script's old name was
`scripts/run_memory_v3.py`, and this folder's old name was `results/memory_v3_run/`).

## Scores

67 questions. Token F1 **0.143**, LLM judge **0.478**. By question type (LLM judge):

| Type | n | LLM judge |
|---|---|---|
| factual | 5 | 0.80 |
| factual_evolving | 17 | 0.65 |
| prediction | 9 | 0.56 |
| transition | 9 | 0.56 |
| recall | 24 | 0.29 |
| pattern_id | 3 | 0.00 |

The full breakdown, including by difficulty, is in `qa_results.json` → `summary`.

## Files

| File | Contents |
|---|---|
| `qa_results.json` | Summary (by type and difficulty) and every question's response, F1 and judge verdict. The inspector UI's Synthetic Eval view reads this file. |
| `ingest_stats.json` | Per-day ingestion report: records extracted, added, duplicates skipped, chunks indexed. |
| `memories.json` | Dump of every preference row in the store after ingestion. |
| `store/` | The memory store (`memories.db`, FAISS indexes). The inspector UI opens it by default. |
