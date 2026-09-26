# LoCoMo: memory_v3 store, single-pass read

**Produced by:** `benchmarks/locomo/run_v3_onepass.py --conversations 1` (defaults:
`--k-chunks 10 --k-summaries 2 --store-run v3_agentic`). It reads the store in
`../v3_agentic/stores/conv-26/` and does not re-ingest, so its memory is identical to
the agentic run's. Only the reading differs.

**Per question:** one semantic search over conversation chunks (top 10), one over
summaries (top 2), all preference rows, then one LLM call with `ANSWER_SYSTEM` (the same
prompt as the mem0 run). Gemini 3.5 Flash.

**Scope:** conversation conv-26, all 199 questions.

## Scores (paper token F1)

| Category | n | F1 |
|---|---|---|
| single-hop | 70 | 0.617 |
| temporal | 37 | 0.634 |
| multi-hop | 32 | 0.422 |
| commonsense | 13 | 0.213 |
| adversarial | 47 | 0.809 |
| **overall** | 199 | **0.608** |
| overall excluding adversarial | 152 | 0.546 |

Cost: 1 LLM call and about 4.4s per question; about 12,400 characters of context on average.

## Files

| File | Contents |
|---|---|
| `locomo_results.json` | Run settings, summary scores, and every record. |
| `conv-26_trace.jsonl` | One line per question: question, gold answer, response, score, everything retrieved (chunks, summaries, preferences), `context_chars` and `seconds`. |
