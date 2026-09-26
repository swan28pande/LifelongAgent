# Inspector app

A browser UI for looking at memory stores, datasets and benchmark results, and for
chatting with a store.

| Path | What it is |
|---|---|
| [backend.py](backend.py) | FastAPI server on port 8000. Opens one memory_v3 store read-only and serves stats, preferences, chunks, summaries, the synthetic dataset and QA results, LoCoMo data and results, and a `/api/chat` endpoint. |
| [frontend/](frontend/) | React + Vite UI with four tabs: Benchmarks, Datasets, Store, Chat. |

## Which data it shows

- **Store:** the newest `results/**/store/` (current runs before `results/legacy/`), which
  is normally `results/synthetic/v3/store`. Pin one with `STORE=<path>`.
- **Synthetic QA:** the `qa_results.json` next to that store.
- **LoCoMo results:** `results/locomo/v3_agentic/locomo_results.json`. Pick another run
  with `LOCOMO_RUN=<folder>`, e.g. `LOCOMO_RUN=mem0`.

```bash
venv/bin/python app/backend.py
STORE=results/legacy/gemini_flash_run/store LOCOMO_RUN=mem0 venv/bin/python app/backend.py
cd app/frontend && npm install && npm run dev
```
