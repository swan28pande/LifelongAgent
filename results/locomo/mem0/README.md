# LoCoMo: mem0 baseline, single-pass read

**Produced by:** `benchmarks/locomo/run_mem0.py --conversations 1` (the script's old name
was `scripts/run_locomo_mem0.py`, same code). mem0 open-source 2.2.0, Gemini 3.5 Flash on
Vertex for extraction and answering, `nomic-embed-text-v1` embeddings, local FAISS,
top-10 retrieval.

**Scope:** conversation conv-26, all 199 questions.

## Scores (paper token F1)

| Category | n | F1 |
|---|---|---|
| single-hop | 70 | 0.464 |
| temporal | 37 | 0.617 |
| multi-hop | 32 | 0.348 |
| commonsense | 13 | 0.174 |
| adversarial | 47 | 0.894 |
| **overall** | 199 | **0.557** |
| overall excluding adversarial | 152 | 0.452 |

mem0's overall lead over v3 comes almost entirely from the adversarial category. It
answers "I don't know" very often (42/47 adversarial, but also 15/70 single-hop and
9/13 commonsense), and the paper metric rewards abstention on adversarial questions.

Ingestion: 19 sessions produced 115 memories (about 6 per session) in 450s. Answering took
about 3.4s and one LLM call per question.

## Files

| File | Contents |
|---|---|
| `locomo_results.json` | Summary scores, ingestion stats, and every record. |
| `conv-26_trace.jsonl` | One line per question: question, gold answer, response, score, and the memories retrieved (text, date, retrieval score). |
| `stores/conv-26/` | mem0's store: FAISS index of memories, plus `history.db` (SQLite audit log of every ADD/UPDATE/DELETE and the raw messages). |

## Setup note

mem0 caps LLM output at 2,000 tokens by default. On Gemini 3.x that budget includes
thinking tokens, which truncated the extraction JSON and silently produced no memories.
The runner sets `max_tokens: 32000` and flags any session that extracts nothing.
