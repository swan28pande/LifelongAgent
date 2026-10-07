# Mem0 monthly benchmark

Run `python -m setup_2.run --method mem0` from the project root. The
[baseline guide](../README.md) covers installation, parallel users and resume.

`method.py` is adapted from the existing `experiments/methods/mem0_oss.py` benchmark
implementation (open-source `mem0ai==2.2.0`). Mem0 2.x extracts facts from each session with
the selected Gemini model and appends them (exact duplicates skipped); it does not UPDATE or
DELETE existing memories, unlike the ECAI 2025 paper's algorithm. Mem0 OSS rejects
`add(timestamp=...)`, so its extraction prompt's Observation Date (the anchor for "yesterday",
"last week") would silently be today's system date; the adapter passes each session's date
instead (`experiments/methods/_common.mem0_observation_date`), as the hosted platform does.
The model is configured with a 32,000-token output limit because Gemini's thinking tokens
otherwise truncate Mem0's JSON. Embeddings use the shared Nomic model and FAISS.

Sessions are added once each, in date order, as they arrive; Mem0 adds its memories
incrementally, so finalization builds nothing. Observed sessions persist in
`store/baseline.sqlite3`, and Mem0's FAISS index and history database in `store/mem0/`.
A session is recorded only after Mem0 accepts it (three attempts with backoff), and an
interrupted month is recovered by the monthly runner's whole-store rollback before replay.

Questions retrieve the ten most similar memories for the user and answer with the existing
common answer prompt, model and original probe-date prefix. Embedding calls are serialized,
since concurrent answer threads sharing one Nomic model corrupt each other's tensors.
Mem0's own extraction calls happen inside the library and are not counted in usage totals.
