# TiMem monthly benchmark adapter

Run `python -m setup_2.run --method timem` from the project root. Scheduling,
cumulative questions, original probe dates, judge rules and reports are shared
with memory_v4. See the [baseline run guide](../README.md) for commands.

`source/` was copied from the local `baselines/TiMem/` checkout, preserving its
native generator, prompts, workflows, documentation and [license](source/LICENSE).
The adapter follows the paper's memory and recall; the profile `timem_complexity_aware`
is recorded in manifests and summaries.

- **Memory.** Every two turns form an L1 fragment (an odd final turn is its own fragment),
  written with the session's three previous fragments as context. L2 session, L3 daily,
  L4 weekly and L5 monthly summaries are written from their children plus the three most
  recent earlier summaries of the same level, as in the native generation workflow.
- **Recall.** The native planner labels each question simple/hybrid/complex and extracts
  keywords. L1 fragments are ranked by 0.9 x normalized dense similarity (top 40) +
  0.1 x normalized BM25 over all fragments, the top 20 are kept in time order, and parents
  are collected bottom-up for the strategy's layers and limits
  (`source/config/datasets/default/retrieval_config.yaml`). The native memory refiner then
  gates the candidates with its strategy prompt before the shared answer prompt.
- **Cost.** Each question makes three model calls: planner, refiner and answer.

Storage is local (SQLite plus in-process Qdrant) instead of the native PostgreSQL/Qdrant
services. The native config manager resolves paths from the working directory, so the
adapter pins `TIMEM_RETRIEVAL_CONFIG_PATH` to the source's recall configuration; without it
TiMem silently falls back to built-in strategies with different layers and limits.
`validation.json` records a check of the earlier dense top-k implementation.

## Monthly state and recovery

All observations, generated nodes and float32 embedding vectors are saved in
`store/baseline.sqlite3`. Each adapter owns an in-process Qdrant collection,
rebuilt from saved vectors on restart without embedding or generation calls.
No Qdrant server or PostgreSQL service is required. Because all durable state
lives inside the store, the monthly runner's backup/rollback includes the entire
hierarchy and vector state. Different users and runs cannot share a collection.

Stable node IDs replace summaries instead of adding duplicates. Child-content
signatures skip unchanged summaries; later months leave completed older-month
summaries unchanged. Raw session ingestion is idempotent and rejects changed
content for an existing session ID.

Weeks are split at calendar month boundaries. A week spanning March and April
produces separate partial weekly memories for those months, so March's monthly
summary never absorbs April observations. Weekly/monthly prompt ranges end at
their last observed child, including partial final months and leap years.
Questions still see all memory accumulated through their evaluation checkpoint.

## Model and source compatibility

Native generation uses the selected benchmark model at temperature 0.0, through
the same usage counter as answer calls. Each summary gets at most three attempts
with exponential backoff and jitter for timeouts, transport failures and transient
HTTP 408/429/500/502/503/504 responses. Each attempt uses a fresh client with a
120-second request timeout and one Gemini SDK attempt; the native per-layer total deadline
also applies. Interrupted stream content is discarded before retrying. Credential,
permission and invalid-request errors propagate immediately. Exhausted failures
propagate to the existing monthly checkpoint recovery; no partial hierarchy is
evaluated. Native prompts and generated-content validators are unchanged.

Summary clients are closed after each attempt on the user's persistent event loop;
the answer client is closed at the end of the invocation. Native prompt logs are
saved in `store/logs/`.

Compatibility changes are confined to the copied source: lazy LLM adapter imports;
optional generator LLM/log-directory injection; asynchronous Gemini streaming;
configuration cache invalidation that does not import unused providers; streamed chunks
joined verbatim (stripping each chunk had glued words together) with truncated
(`MAX_TOKENS`) or empty Gemini output raised instead of saved; and the recall config's
memory refiner pointed at the Gemini provider and the `memory_refiner_*` prompts that
exist (the shipped `relevance_analysis_*` names do not, which silently fell back to a
generic prompt). Native generation algorithms and prompts are kept.

Install this folder's `requirements.txt`, which covers the benchmark adapter's
minimal dependencies. The larger `source/requirements.txt` remains available
for the native application's additional workflows and services. The benchmark
uses the existing Nomic document/query prefixes and cosine retrieval in Qdrant.

## Local launch

On this machine, the existing environment at
`tmp/memory-v3-update-validation/` contains the required CPU dependencies.
The launcher reuses the downloaded Nomic model in
`tmp/locomo-v3-update-cache/huggingface/` and the current Google application-default
credentials. All temporary files and caches stay inside this repository.

From the project root, run the entire dataset with the existing answer model
`gemini-3.5-flash` and judge `gemini-3.1-pro-preview`:

```bash
bash setup_2/baselines/TiMem/run_local.sh
```

The launcher defaults to five parallel user processes, one answer worker and one
judge request per user. Each user's months remain sequential. Results go to
`setup_2/baselines/TiMem/results/v2_timem_local/`. The complete schedule has 120
user/month checkpoints and 11,721 cumulative answers. Run TiMem by itself initially.

For a single-user first-month check:

```bash
bash setup_2/baselines/TiMem/run_local.sh --users u5 --through 2026-03 \
  --user-workers 1 --run v2_timem_u5_check
```

Arguments override launcher defaults. Set `TIMEM_PYTHON` to use another environment
after installing this folder's requirements. Rerun the same command to resume;
completed checkpoints are skipped, an interrupted build replays that month, and
interrupted evaluation retries missing answers/grades. Use a new run name for
results made with the earlier implementation. Keep the machine awake during a run.

To keep the full run active after closing the terminal:

```bash
nohup bash setup_2/baselines/TiMem/run_local.sh \
  >> setup_2/tmp/timem_local.log 2>&1 < /dev/null &
```

Offline tests cover local Qdrant, all five native levels, callbacks and token usage,
cached-vector reloads, boundaries, bounded retries, discarded partial streams,
monthly recovery and separate user processes. Current local verification evidence
is saved under `setup_2/tmp/timem-local-validation/`.

The [local validation report](validation.json) records 70 passing offline tests.
A real u5 March checkpoint completed all five levels from 13 conversations and
answered and judged all six questions correctly, without retries at the final
deadline setting. Recorded ingestion, finalization, answering and judging time
totaled 713 seconds (11 minutes 53 seconds). Rerunning that completed checkpoint
made no model calls. The subsequent addition of aiohttp/auth transport exception
coverage passed the final offline suite; normal generation and deadline settings
were unchanged. This is a first-month check; the full dataset has not been launched.
