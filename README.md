# Lifelong Agent

A memory system for an assistant that learns a user's evolving preferences over months
of conversation, and a set of benchmarks comparing it with other memory systems.

## Layout

| Folder | What it holds |
|---|---|
| [memory_v3/](memory_v3/) | **Current system.** Ingestion pipeline (extract → consolidate → dedupe → write → index) and a tool-calling read agent. |
| [memory_v2/](memory_v2/) | Storage layer (SQLite + FAISS) and hierarchical summarizer, reused by v3; plus the older v2 agent. |
| [benchmarks/](benchmarks/) | **How to run every evaluation.** One folder per benchmark, one runner per system. |
| [results/](results/) | **Every run's output**, organised as `results/<benchmark>/<system>/`. |
| [datasets/](datasets/) | The synthetic preference-evolution dataset (conversations + QA). |
| [baselines/](baselines/) | Third-party systems and datasets (LoCoMo data, MemGPT, MemoryBank, RSum, MrRec, NaiveRAG). |
| [app/](app/) | Inspector UI: FastAPI backend + React frontend for browsing stores, datasets and results. |
| [evaluation/](evaluation/) | Older memory_v2 evaluators (LoCoMo, MSC, probing, synthetic). Superseded by `benchmarks/`. |
| [scripts/](scripts/) | Dataset generation and older memory_v2 pipeline runs. |
| [tests/](tests/) | Offline capability tests for memory_v3. |
| [related_papers/](related_papers/) | PDFs of the papers this work builds on. |

## Quick start

```bash
# Synthetic benchmark (single user, 60 days of preference changes)
venv/bin/python benchmarks/synthetic/run_v3.py

# LoCoMo, one conversation, each system
venv/bin/python benchmarks/locomo/run_v3_agentic.py --conversations 1
venv/bin/python benchmarks/locomo/run_v3_onepass.py --conversations 1   # reads the agentic run's store
venv/bin/python benchmarks/locomo/run_mem0.py       --conversations 1

# Inspector UI
venv/bin/python app/backend.py            # API on :8000
cd app/frontend && npm run dev            # UI
```

Models run on Gemini through Vertex AI. Authenticate once with
`gcloud auth application-default login`.

Where the current numbers stand: [results/README.md](results/README.md).
