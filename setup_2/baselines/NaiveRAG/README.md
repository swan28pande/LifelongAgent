# NaiveRAG monthly benchmark

Run `python -m setup_2.run --method naive_rag` from the project root. The
[baseline guide](../README.md) covers installation, parallel users and resume.

`naive_rag.py` remains an unchanged copy of the original
`baselines/NaiveRAG/naive_rag.py`, exposing `ingest_sessions()`, `search()`
and `reset()`. The monthly runner uses `method.py`, adapted from the existing
`experiments/methods/naive_rag.py` benchmark implementation.

Observed sessions persist in `store/baseline.sqlite3`. At finalization, only
new sessions are chunked into five-turn documents and added to the FAISS index,
which persists in `store/faiss/`. SQLite metadata records the indexed session
IDs and chunk count, allowing a restart to reload the index without embedding
old documents. An interrupted index write is recovered by the monthly runner's
whole-store rollback before replaying that month.

Questions retrieve the ten closest raw chunks, sort them chronologically, and
use the existing common answer prompt, model and original probe-date prefix.
The Nomic embedding model keeps the existing document/query prefixes. Nothing
is extracted or summarized; evaluation questions and judge feedback are never
added to memory. Empty histories can be evaluated and return an empty context.

Duplicate identical sessions are ignored; changed content for an already stored
session ID is rejected. Only the current run's own FAISS files are reloaded.
Finalization must finish before questions are answered.

Install this folder's `requirements.txt`. No database server is needed.
Offline tests use actual FAISS with deterministic embeddings and a fake model,
covering successive months, index reload, partial-build rollback, answer/judge
resume and independent user processes. No live model benchmark was run.
