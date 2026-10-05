# Monthly cumulative baseline evaluation

Status: complete

Run TiMem, NaiveRAG, and DirectPrompting/full context on canonical `datasets/v2/`
using the same monthly checkpoints, cumulative questions, original probe dates,
models, answer formatting, judge, recovery barriers, and metrics as memory_v4.
Implementation and runtime artifacts belong in `setup_2/`. Existing agents,
original baselines, shared experiment code, and datasets remain unchanged.

## Required behavior

- Ingest only the current month's observations, chronologically, into each user's
  persistent store. Never pass future sessions or evaluation questions to memory.
- Finalize the current store and answer every question introduced by that
  checkpoint, preserving its original probe date and reference metadata.
- Complete answers and judging before advancing that user's next month.
- Separate stores/results by method, run, and user. Reload built checkpoint memory
  after interruptions without re-running completed summaries or model calls.
- Roll back an incomplete memory build as a unit, including vectors and summaries.
- Preserve the existing memory_v4 default and allow independent user processes.

## Implementation decisions

- Reuse the existing monthly runner, loader, grading, and reporting. Add explicit
  method selection and optional process parallelism across users.
- Baseline observations are durable in SQLite. NaiveRAG saves its FAISS index;
  full context reloads raw sessions and retains all observed history, failing
  explicitly if its configured approximate budget is exceeded.
- TiMem uses its copied native generator and prompts, the shared benchmark LLM,
  persistent generated nodes/vectors in SQLite, and an isolated in-process Qdrant
  search index rebuilt from cached vectors on construction. No database service
  or remote collection state is required for this benchmark adapter.
- TiMem updates changed parents only, with stable node identifiers. Weekly ranges
  are split at calendar month boundaries so a monthly summary never absorbs a
  neighboring month's observations; partial-period bounds stop at observed data.
- Preserve the existing TiMem benchmark adapter's dense top-k retrieval profile,
  explicitly distinguish it from native TiMem complexity-aware retrieval, and
  publish that distinction with run metadata and documentation.
- Native generator retries are bounded by per-layer timeouts; errors propagate to
  the runner, which leaves the checkpoint resumable instead of hanging forever.
- Native async generation keeps one event loop per user's invocation so pooled
  model transports remain valid; async clients are closed on that loop.

## Validation

Use actual FAISS/Qdrant locally with deterministic LLM and embedding substitutes.
Cover cumulative schedules, original-date interpretation, chronological barriers,
restarts, interrupted builds/answers/judging, user/run/method isolation, calendar
boundaries, empty months, and token accounting. Exercise the native TiMem generator
with a fake chat model. Preview all current users and baseline CLIs without models
or paid calls; verify dataset and original source hashes remain unchanged.

## Completion

All three methods are available through `python -m setup_2.run --method ...`.
The default memory_v4 behavior and result location are preserved. Each baseline
has its own output directory, durable store, recovery behavior and run guide.
Optional spawned user processes reuse the same per-user sequential protocol.

51 offline tests passed in 11.47 seconds with no skips, using actual FAISS/Qdrant,
native TiMem generation against a fake async model, simulated loop-bound provider
transports, timeout cancellation, recovery and parallel user processes. Native
thread/event-loop tests needed IPC availability outside the restrictive sandbox.
All four CLI schedules match: five users, 24 checkpoints per user, 2,738 sessions,
1,000 distinct questions and 11,721 cumulative answer instances per method.
All 4,055 protected original files retain their hashes. No paid calls were made.

Evidence: `validation.json` here and
`setup_2/tmp/baseline-monthly-validation/`. The baseline run guide is
`setup_2/baselines/README.md`.
