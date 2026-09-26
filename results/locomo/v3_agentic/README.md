# LoCoMo: memory_v3, agentic read

**Produced by:** `benchmarks/locomo/run_v3_agentic.py --conversations 1` (summaries on).
This run predates the folder reorganisation. It was made by the same code under its
old name, `scripts/run_locomo_v3.py --summaries`, with the `semantic_search_summaries`
tool available to the agent.

**Scope:** conversation conv-26 (Caroline & Melanie), all 199 questions, Gemini 3.5 Flash.

## Scores (paper token F1)

| Category | n | F1 |
|---|---|---|
| single-hop | 70 | 0.667 |
| temporal | 37 | 0.605 |
| multi-hop | 32 | 0.394 |
| commonsense | 13 | 0.222 |
| adversarial | 47 | 0.149 |
| **overall** | 199 | **0.460** |

Cost: 1,388 tool calls in total, about 7 per question and about 30s per question.

## Scores judged by meaning (manual, see `manual_judgement.jsonl`)

| Category | Correct | Partial | Wrong | Accuracy |
|---|---|---|---|---|
| single-hop | 67 | 1 | 2 | 96% |
| temporal | 32 | 0 | 5 | 86% |
| multi-hop | 25 | 4 | 3 | 78% |
| commonsense | 10 | 0 | 3 | 77% |
| adversarial | 32 | 6 | 9 | 68% |
| **overall** | 166 | 11 | 22 | **83%** |

Main real errors: "last Friday"/"last Tuesday" resolved one week too early (Q29, Q30, Q42);
internal reasoning leaking into the answer when the tool loop ran long (Q54, Q127); empty
responses (Q78, Q167, Q181); accepting a false premise on some adversarial questions.

## Files

| File | Contents |
|---|---|
| `locomo_results.json` | Summary scores, tool-call statistics, ingestion stats, and every record. |
| `conv-26_trace.jsonl` | One line per question: question, gold answer, response, score, and the full tool-call sequence with arguments. |
| `manual_judgement.jsonl` | Per-question verdict (correct / partial / wrong) judged by meaning, with a note on each error. |
| `summary_audit.txt` | For every question where the agent called `semantic_search_summaries`: the query it used and the summaries that came back, to check whether summaries contain the answer. |
| `stores/conv-26/` | The memory store built during ingestion: `memories.db` (preference rows, conversation chunks, summary metadata), `faiss_conversations/`, `faiss_summaries/`. `run_v3_onepass.py` reads this store. |
