# memory_v2: storage layer and the previous system

memory_v3 reuses `store.py` and `summarizer.py` from here. The rest is the earlier v2
agent, kept because its results are in `results/legacy/`.

| File | What it does | Used by v3? |
|---|---|---|
| [store.py](store.py) | `MemoryStore`: SQLite (`memories` preference rows, `conversations` chunks, `summary_meta` build hashes) plus FAISS indexes for conversation chunks and summaries, with `nomic-embed-text-v1` embeddings. | yes |
| [summarizer.py](summarizer.py) | Hierarchical summaries: weekly → monthly → yearly → lifetime, per speaker. `run()` builds everything; `update_after_ingest(date)` rebuilds only the chain whose source data changed. | yes |
| [agent.py](agent.py) | `LifelongAgent`, the v2 interface (extract → store → summarize → build context → answer). | no |
| [extractor.py](extractor.py) | v2's LLM preference extractor. | no |
| [context_builder.py](context_builder.py) | v2's fixed context assembly (lifetime summary + recent summaries + retrieved chunks). | no |
| [evaluate.py](evaluate.py) | v2's QA evaluation over a store (defaults to `results/legacy/gemini_extraction_run/store`). | no |
| `data/` | Default v2 store location. | no |
