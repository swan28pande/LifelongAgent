# Reliable local TiMem monthly run

Status: complete

## Required behavior

Prepare only the TiMem baseline in `setup_2` for a local `datasets/v2` run and
provide one reproducible launch command. Keep the monthly ingestion barriers,
cumulative questions, original probe dates, five native summary levels, dense
retrieval profile, selected models, judge, and other baselines unchanged.

Summary calls must have bounded deadlines and bounded retries for transient
transport/API failures, including failures after streaming begins. Failed partial
streams must not be saved as completed memories. Permanent credential/request
failures must propagate promptly. Preserve monthly checkpoint recovery.

Install or verify local dependencies, credentials, and the Nomic cache. Keep
temporary files and caches inside this repository. Verify recovery offline and
complete a bounded real first-month run with the existing LLM judge; do not launch
the full benchmark on the user's behalf.

## Scope

TiMem-specific adapter, requirements, local launch instructions/scripts, and
focused tests/artifacts only. Do not edit datasets, `memory_v4`, NaiveRAG,
DirectPrompting, or the common scheduling/judging protocol.

## Verification

- Local dependencies are installed and compatible; cached Nomic embeddings and
  Google ADC refresh were verified.
- Final offline suite: 70 passed in 10.82 seconds, including actual aiohttp/httpx
  stream interruption, bounded exhaustion, fatal credential errors and user isolation.
- Real u5/March checkpoint: 13 sessions, L1–L5 complete, six answers and six LLM
  grades, 6/6 correct; no retries with the final 120-second request deadline.
- Completed-checkpoint resume made no model calls and changed no user artifacts.
- All 3,652 protected files outside TiMem/tests remain unchanged.
- Full local command: `bash setup_2/baselines/TiMem/run_local.sh`; five user
  processes, one answer worker and one judge request per user. No full run started.

Evidence: `setup_2/baselines/TiMem/validation.json` and the linked logs/results.
The report identifies the live revision and the later offline-tested transport
exception additions; native prompts, models, deadlines and successful execution
are unchanged by those additions.
