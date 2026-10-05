# Gemini concurrency measurements

The user requested a bounded live test of whether three baseline evaluations
retain throughput when their Gemini traffic overlaps. API calls are authorized.
The existing agents, baseline implementations, loader, prompts, judge and datasets
are unchanged. This is a throughput pilot, not a baseline accuracy evaluation.

## Controlled first-month comparison

`pilot.py` loads canonical user u5's first checkpoint, March 2026: all 13 observed
sessions and all six scheduled questions. Separate TiMem, NaiveRAG and full-context
stores use the actual implementations. TiMem builds its native five-level summaries
with the configured Gemini model; both retrieval methods use the actual Nomic
embedder and local vector stores. Preparation and one-answer/one-judge warmups per
method are recorded separately from measurements.

Each method answers the same six questions alone and with all three methods
overlapping. The original shared Pro judge and judging prompt are used. Judging
uses frozen responses from the first solitary answer condition so its input is
identical between comparisons. Two rounds reverse solitary/concurrent ordering;
solitary method order also reverses in the second round.

The configured answer/judge limits are 8/16 per method, matching production defaults.
With only six questions per method, effective measured concurrency is six alone
and at most 18 together, for both answering and judging. The pilot cannot establish
capacity at 120 concurrent answers or 240 judgments. Within each timed stage,
requests start together rather than letting construction times stagger the methods.

## Late-history full-context stress

`late_context.py` feeds all of u5's observed history into a separate full-context
store, preserving monthly ingestion order, and selects four of its latest scheduled
questions. It reopens the first-month retrieval stores and uses their original
first-month questions. Each stream is measured alone and with all three overlapping,
in two rounds with reversed order. This tests contention from much larger full-context
prompts while retrieval prompts stay small. Histories deliberately differ across
streams, so these results must not be interpreted as an accuracy comparison or as
an end-to-end late-checkpoint run of all three methods.

Effective concurrency is four per method and at most 12 combined. The existing
judge is already measured in the first-month comparison and is not repeated here.

## Instrumentation and interpretation

- Actual credentials resolve the Google Cloud project; manifests record project,
  quota project, endpoint location, model names, question identifiers, dataset hashes
  and source hashes. Credentials and authorization headers are never recorded.
- Per-client wrappers observe the Google Gen AI SDK's `_request_once` and
  `_async_request_once`, leaving transports, prompts and retry policies unchanged.
  A failed HTTP attempt is recorded before the SDK retries it. Streaming successes
  record accepted request headers; logical latency includes the complete answer.
- Callback metadata records actual input/output, reasoning and cached input tokens.
  Output counts follow provider usage metadata; reasoning is recorded separately
  as a reported component, not added again to output tokens.
- Logical request latency includes retrieval and network/model work after thread or
  semaphore admission. These already warmed retrieval steps are small. Each method's
  active batch span can be recovered from completion timestamps minus logical
  durations; aggregate stage spans come from the saved phase records.
- Requests/minute and input tokens/minute are effective throughput over short batches,
  not a measured sustained quota or a promise about available capacity.
- All three streams execute locally in one process with independent stores and model
  clients. They use the VM's Vertex AI project. This isolates API contention from
  the current VM's 2-vCPU/4-GB constraints and does not validate 15 spawned processes.
- Two repetitions and small question sets provide operational observations, not a
  statistical claim of equal performance. Absence of 429 errors does not prove that
  higher concurrency or long sustained traffic will avoid throttling.

## Five-user-equivalent API replay

`scaled_api.py` captures the exact messages produced by the actual first-month
baseline retrieval and answer formatting. Capture returns an empty placeholder
solely to intercept messages; it is never treated as an answer. All measured
answer requests call the real selected model with those messages. Judging reuses
the previously saved real answers and original grading code.

The first-month six-question workloads are repeated to fill five user-equivalent
request slots per method: 40 concurrent answers or 80 concurrent judgments. Each
method is measured alone before all three overlap, reaching at most 120 answer
calls or 240 judge calls. One comparison round makes at most 720 measured logical
requests plus warmups; exhausted final failures stop subsequent higher load.

This profile isolates API contention from retrieval CPU and model copies. Its
replicas do not stand in for five different biographies, 15 real memory stores,
sustained traffic or late-history full-context prompts. Repeated prompts can also
benefit from provider caching; actual cache metadata is retained. Use the separate
late-history test to assess the effect of larger prompts.

## Preparation interruption

The initial preparation run `u5_march_20261004_a` encountered one 180-second TiMem
L3 timeout after 31 accepted HTTP attempts. No HTTP throttling response preceded
that timeout. Its raw log and incomplete run summary are retained. The resumed
pilot copies only the completed preparation stores into a fresh run directory;
native node signatures skip completed summaries. The comparison measurements
remain separate from that preparation failure. `pilot_initial.py` preserves the
initial harness before the resume option was added.

## Reproduction

Use the existing project-local validation environment and ADC. The real Nomic model
cache lives inside the project. A missing `langchain` package was installed into
that environment because importing `memory_v3.store` also imports its agent package.
Missing small Nomic configuration files were downloaded into the existing cache.

```bash
mkdir -p setup_2/tmp
export TMPDIR="$PWD/setup_2/tmp"
export HF_HOME="$PWD/tmp/locomo-v3-update-cache/huggingface"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
tmp/memory-v3-update-validation/bin/python -u \
  .research/gemini-concurrency-pilot/pilot.py --run NEW_FIRST_MONTH_RUN
tmp/memory-v3-update-validation/bin/python -u \
  .research/gemini-concurrency-pilot/late_context.py \
  --source-run NEW_FIRST_MONTH_RUN --run NEW_LATE_HISTORY_RUN
tmp/memory-v3-update-validation/bin/python -u \
  .research/gemini-concurrency-pilot/scaled_api.py \
  --source-run NEW_FIRST_MONTH_RUN --run NEW_SCALED_API_RUN
```

Use fresh run names for independent measurements. Raw `events.jsonl`, provider judge
logs, phase summaries, manifests and overall runtimes are kept in the run directories.
The measurement summary and its interpretation are generated separately from this
raw evidence.
