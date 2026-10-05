# Monthly cumulative evaluation

This setup evaluates the unchanged `memory_v4` agent and the three
[baseline methods](baselines/README.md) by growing their stores one calendar month
at a time. The default uses the existing `OursV4D` adapter, agent/provider
configuration, embedding model, five-tool retrieval loop, and common LLM judge.
Both hierarchical summaries and distillation are enabled at every checkpoint.
The agent and shared evaluation runner are unchanged. This loader reads the supplied
dataset and never repairs or rewrites it.

The accepted question interpretation is recorded in [DECISIONS.md](DECISIONS.md).
Every earlier question is repeated each month with its **original probe date,
reference, and accepted answers**. Thus "current" in a March question still means
March when repeated in April. Newly introduced April questions can test April's
current state. The database holds all conversations observed through April.

## Monthly sequence

1. Add only this month's sessions, in date order, to the user's existing database.
2. Finalize that method's memory. The default memory_v4 finalization remains
   `build_summaries(force=True, distill=True)`; baseline finalization updates its
   index, hierarchy or complete-history prompt.
3. Answer all questions introduced up to this checkpoint. Use the original
   `(Asked on YYYY-MM-DD)` prompt prefix; the evaluation checkpoint is logged separately.
4. Grade with the existing judge prompt and accepted-answer/string checks.
5. Save the completed checkpoint before ingesting the next month's sessions.

Different users have separate stores. `--user-workers` enables independent user
processes and defaults to one. Answering uses the existing default of eight
workers against the fixed checkpoint store; judging uses the existing concurrency
of sixteen. Months and ingestion sessions are sequential. Evaluation questions,
reference answers, responses, and judge feedback are never ingested.
One agent is reused across months for each user during an invocation; resume reloads
that user's persisted database before continuing.

All calendar months in the history are processed, including months without new
probes or sessions and the final partial month. The repaired dataset introduces new
questions on February 28, 2028, alongside repetitions of the January cumulative set.
Scheduled retrospective IDs remain separate
questions; the loader does not deduplicate by `original_id`.

## Commands

Run from the repository root with the same Python environment and provider
credentials used for the existing memory_v4 evaluation. No new agent dependencies
are introduced. Defaults are the shared `experiments/config.py` settings:
`gemini-3.5-flash` for memory construction/answers, `gemini-3.1-pro-preview` for
the judge, and the unchanged nomic embedding model in the agent.

Inspect the schedule without loading models, accessing credentials, or creating
run files:

```bash
python -m setup_2.run --dry-run
python -m setup_2.run --users u1 --through 2026-04 --dry-run
```

Run one user through April, then continue that same database through the remaining
months by dropping `--through`:

```bash
python -m setup_2.run --users u1 --through 2026-04 --run monthly_u1
python -m setup_2.run --users u1 --run monthly_u1
```

Run all users, optionally in five independent processes:

```bash
python -m setup_2.run --run monthly_all
python -m setup_2.run --run monthly_parallel --user-workers 5
```

The canonical input is `datasets/v2/`. `--data-dir` selects another copy. Use a new
run name after correcting dataset files or changing the selected users/models/code;
the run manifest checks input and implementation hashes and refuses to mix different evaluations.
Worker counts and `--through` may change when resuming. The loader does not repair
or silently exclude semantic dataset errors; grading uses the supplied references.

Results and temporary files are confined to this setup. Runtime `TMPDIR` points
to `setup_2/tmp/`. Hugging Face caching defaults to `setup_2/.cache/huggingface/`
unless an existing `HF_HOME` is configured. Provider credentials are reused in place.

`--method timem`, `--method naive_rag`, and `--method full_context` select the
baselines using this same protocol and judge. Their dependencies, commands,
method definitions and output locations are in the [baseline guide](baselines/README.md).
Each baseline writes into its own `setup_2/baselines/<folder>/results/<run>/`.
Omitting `--method`, or selecting `ours_v4d`, retains the original output location
and unchanged memory_v4 agent. The runner now records the selected method/profile
and implementation hashes, including its baseline code and native TiMem prompts.
Results from the earlier runner revision require a new run name; they are not
silently mixed with this implementation.

## Results and resume

```text
setup_2/results/<run>/
  manifest.json                    protocol, selected input hashes, models
  summary.json                     overall and per-month/per-user scores
  <user>/
    store/                         growing SQLite and FAISS memory
    state.json                     completed checkpoints and active phase
    checkpoint_backup/             temporary undo for unfinished memory construction
    months/<YYYY-MM>/
      ingest.jsonl                 this month's successful ingestion records
      answers.jsonl                original question dates, references, responses, tools
      grades.jsonl                 judge verdicts, strict checks, judge tokens
      judge_log.jsonl              existing judge's call/usage log
      usage.json                   agent tokens and stage wall time, including retries
      summary.json                 cumulative/new/repeated/category scores and latency
```

Each checkpoint has its own answer/grade files, so repeating a question is a new
evaluation rather than a skipped question ID. The overall count is **answer
instances across checkpoints**, not the number of distinct question IDs. Headline
accuracy remains LLM-judge accuracy; strict checks are retained in grade records.

Rerun the same command after an interruption. Completed checkpoints are validated
and skipped. If memory construction was interrupted, the runner restores the last
checkpoint's store and **rebuilds only the unfinished month**, including summaries
and distillation. SQLite and FAISS do not share a single transaction, so continuing
an interrupted session's partial writes would risk duplicates or inconsistent memory.
One backup is retained while building, then removed once evaluation can resume safely.

If answers or grading were interrupted, the built checkpoint memory remains fixed
and only missing answers/grades are retried. Raised call failures stop advancement;
they are not manufactured as incorrect answers. Normal agent abstentions remain
answers. A run lock prevents concurrent processes from changing the same store.
Missing completed results or mismatched manifests cause an explicit error rather
than replaying an old checkpoint against a database containing later months.

Timing and model usage survive ordinary exceptions; an abrupt process kill may
omit timing/tokens from the operation in flight. Completed records remain durable.
Judge-call logs retain successful and failed attempts. This is a runner, not a
background process manager; launch it under the desired session manager for unattended use.

With parallel users, aggregate stage seconds sum work across users and can exceed
elapsed wall time. `seconds_this_invocation` records elapsed wall time for the
invocation, and the per-checkpoint usage files retain construction, answering and
judging time across resumes.

## Offline verification

```bash
mkdir -p setup_2/tmp
TMPDIR="$PWD/setup_2/tmp" PYTHONDONTWRITEBYTECODE=1 python -m pytest setup_2/tests -q \
  -o cache_dir=setup_2/tmp/pytest-cache --basetemp=setup_2/tmp/pytest
```

Tests use persistent SQLite-backed memory and a deterministic judge, plus an adapter
check that verifies the existing summaries/distillation finalization. They verify
cumulative scheduling, original-date interpretation, future-data exclusion, independent
user stores, partial-month handling, resume barriers, interruption rollback, and
manifest/result integrity. They make no external model calls.

Verified on October 4, 2026: **51 tests passed** with no skips, using the existing
`tmp/memory-v3-update-validation/bin/python` environment. An offline schedule check
on all five current users accounts for 2,738 sessions, 1,000 distinct question IDs,
24 checkpoints per user, and 11,721 answer instances after cumulative repetition.
The u1 March/April preview has 12 questions in March and 19 in April. Source-hash
and Git checks confirmed that existing input and implementation files were unchanged.
These are loader/control-flow checks; no live model evaluation was run.

The additional baseline tests exercise real FAISS and Qdrant with fake models and
embeddings, native TiMem generation/callback accounting, calendar boundaries,
durable history, partial-build rollback, answer/judge resume and spawned user
processes. Run these tests in an environment that permits thread/event-loop and
process IPC; a restrictive sandbox can stall those operations.
