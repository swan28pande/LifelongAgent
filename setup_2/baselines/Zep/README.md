# Zep monthly benchmark (self-hosted Graphiti)

Run `python -m setup_2.run --method zep` from the project root. The
[baseline guide](../README.md) covers installation, parallel users and resume.

Zep Cloud is not available to this project, so `method.py` runs Graphiti
(`graphiti-core==0.30.2`), the open-source temporal knowledge-graph engine behind Zep, on an
embedded Kuzu database at `store/graph.kuzu`. Each session is one message episode dated at the
session's date, with user turns labelled by the user's name. Graphiti extracts entities and
facts, resolves them against the existing graph, and gives each fact a validity interval,
invalidating facts that a later episode contradicts. Zep Cloud ingests one message per
episode; one episode per session keeps a two-year history affordable (about a minute and ten
LLM calls per session).

Every Graphiti call uses the selected Gemini model with a 32,768-token output limit. The
adapter overrides three Graphiti defaults that would otherwise change results silently: some
prompts go to `gemini-2.5-flash-lite`, edge extraction is capped at 16K output tokens (thinking
included), and truncated JSON is partially salvaged instead of retried. It also creates Kuzu's
full-text indexes, which `KuzuDriver` never builds; without them BM25 search fails and only
logs an error. Embeddings use the shared Nomic model; reranking uses the local
`BAAI/bge-reranker-v2-m3` cross-encoder.

Questions retrieve as in the Zep LoCoMo evaluation used for the paper: 20 facts (BM25 + cosine,
cross-encoder reranking) and 20 entities (BM25 + cosine, RRF), in Zep's FACTS/ENTITIES context.
Each fact shows its event time and, when invalidated, the date it stopped being true. Answers
use the common answer prompt, model and original probe-date prefix.

A session is recorded in `store/baseline.sqlite3` only after Graphiti stores its episode (four
attempts with backoff). Finalization checkpoints Kuzu's write-ahead log so the monthly
whole-store backup copies a consistent database. All Graphiti LLM calls, thinking tokens
included, are counted in usage totals.
