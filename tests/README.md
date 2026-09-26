# tests

| File | What it checks |
|---|---|
| [test_memory_v3.py](test_memory_v3.py) | Capability tests for memory_v3. Runs offline with no API keys: a scripted stub replaces the LLM and a deterministic fake replaces the embedding model, so it checks the plumbing (what gets written, rejected entity names, tool schemas), not retrieval quality. |
| [test_ingestion_comparison.py](test_ingestion_comparison.py) | v2 and v3 ingestion side by side on the same input. Offline and deterministic by default; `LIVE=1` runs it against a real model. |

```bash
venv/bin/python tests/test_memory_v3.py
venv/bin/python tests/test_ingestion_comparison.py
```
