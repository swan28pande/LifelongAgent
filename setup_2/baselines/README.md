# Monthly baseline evaluation

TiMem, NaiveRAG, Mem0, Zep (self-hosted Graphiti) and DirectPrompting/full context now run through the same
[monthly runner](../README.md) as memory_v4, using canonical `datasets/v2/`.
Each user ingests one month, finalizes memory, answers all questions introduced so
far, and completes the existing LLM judging before receiving the next month.
Repeated questions retain their original probe date, wording, and references.

| Method flag | Implementation | Persistent memory | Retrieval profile |
| --- | --- | --- | --- |
| `timem` | [TiMem](TiMem/README.md) | Raw sessions, five-level summaries and cached vectors in SQLite; isolated local Qdrant search | Native complexity-aware recall: planner, 0.9 dense + 0.1 BM25 L1, bottom-up parents, memory refiner |
| `naive_rag` | [NaiveRAG](NaiveRAG/README.md) | Raw sessions in SQLite and an incremental FAISS index | Five-turn chunks; dense top-10 retrieval |
| `full_context` | [DirectPrompting](DirectPrompting/README.md) | Raw sessions in SQLite | All observed history directly in the answer prompt |
| `mem0` | [Mem0](Mem0/README.md) | Raw sessions in SQLite; Mem0's extracted facts in FAISS plus its history database | Additive LLM fact extraction (Mem0 2.2.0) grounded in the session date; dense top-10 over extracted facts |
| `zep` | [Zep](Zep/README.md) | Raw sessions in SQLite; Graphiti temporal knowledge graph in embedded Kuzu | One episode per session; entities and facts with validity intervals; 20 facts (hybrid, cross-encoder) + 20 entities (hybrid, RRF) |

The common backbone is `gemini-3.5-flash`, the judge is
`gemini-3.1-pro-preview`, and retrieval uses the existing prefixed Nomic embedder.
Models, dates, answer formatting, grading and result metrics use the existing
experiment configuration and interfaces. TiMem's native generation uses that
same selected backbone and usage callback. Original baseline sources outside
`setup_2/`, agents and datasets remain unchanged.

TiMem here follows the paper's memory and recall, adapted for safe monthly execution:
native 2-turn L1 fragments and level summaries with same-level history, and the native
complexity-aware recall (planner, hybrid L1 ranking, bottom-up parents, memory refiner).
Each question costs three model calls (planner, refiner, answer). Manifests and
summaries identify this profile as `timem_complexity_aware`.

## Install

Use the existing compatible project environment and Vertex AI application-default
credentials. This command installs the requirements for all three methods;
TiMem's benchmark requirements include the shared provider and embedding dependencies.

```bash
mkdir -p setup_2/tmp
export TMPDIR="$PWD/setup_2/tmp"
export PIP_CACHE_DIR="$PWD/setup_2/tmp/pip-cache"
python -m pip install -r setup_2/baselines/TiMem/requirements.txt
```

For a single method, install its own `requirements.txt` instead. The native
source's larger requirements file is for its full application and is not needed
for this benchmark adapter. Qdrant runs in process; database containers are not
required. The Nomic model must be cached or downloadable. Model caching defaults
to `setup_2/.cache/huggingface/`; provider credentials are reused in place.

## Run

Run from the repository root. Preview without models, credentials or run writes:

```bash
python -m setup_2.run --method timem --dry-run
python -m setup_2.run --method naive_rag --dry-run
python -m setup_2.run --method full_context --dry-run
python -m setup_2.run --method mem0 --dry-run
```

Run all five users, with separate processes for users and sequential months
within each process:

```bash
python -m setup_2.run --method timem --run v2_monthly --user-workers 5
python -m setup_2.run --method naive_rag --run v2_monthly --user-workers 5
python -m setup_2.run --method full_context --run v2_monthly --user-workers 5
python -m setup_2.run --method mem0 --run v2_monthly --user-workers 5
```

These are independent commands. Run one method at a time for a controlled
comparison, or give each invocation the resources and quota it needs.
`--workers` defaults to 8 answer threads per user, `--judge-concurrency` to 16
judge calls per user, and `--user-workers` to 1. Counts may change on resume.
Five user processes therefore permit up to 40 concurrent answers or 80 concurrent
judge calls across users at the respective stage.

For a short initial run and its continuation:

```bash
python -m setup_2.run --method timem --users u1 --through 2026-04 --run v2_u1
python -m setup_2.run --method timem --users u1 --run v2_u1
```

The same options work for NaiveRAG and full context. Rerun the same command after
an interruption. User selection, inputs, models and method implementation must
match the run manifest; use a new run name when changing them.
An unfinished memory build restores the previous store before replaying that
month. Interrupted answers/judging reuse the built memory and retry only missing
results. Completed checkpoints make no new model calls.

## Results

```text
setup_2/baselines/TiMem/results/<run>/
setup_2/baselines/NaiveRAG/results/<run>/
setup_2/baselines/DirectPrompting/results/<run>/
```

Each contains a manifest, overall summary, and independent per-user stores,
checkpoint state and monthly ingestion/answer/judge/usage/summary files. The
default memory_v4 command still writes to `setup_2/results/<run>/`.
Do not reuse results from the earlier runner revision; implementation hashes
intentionally prevent combining different code versions.

The current dataset schedules 24 checkpoints per user, 2,738 sessions, 1,000
distinct questions and **11,721 cumulative answer instances per method**.
Every user's full history fits the existing approximate full-context budget.
These counts come from offline schedule inspection, not benchmark results.

Offline verification covers actual FAISS/Qdrant stores, native TiMem generation
with a fake model, callback token accounting, monthly barriers, checkpoint
recovery and spawned user processes. **51 tests passed** on October 4, 2026.
The saved [validation report](validation.json) includes verified dataset hashes
and library versions. Schedule previews are in
`setup_2/tmp/baseline-monthly-validation/`.
No paid benchmark was started.
