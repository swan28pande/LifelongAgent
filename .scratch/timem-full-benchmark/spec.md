# Full local TiMem monthly benchmark

Status: running

Run the existing TiMem adapter in `setup_2/` on all five users in
`datasets/v2/`, using the existing monthly cumulative protocol and LLM judge.
Users run in five independent processes; checkpoints remain sequential within
each user. Repeated questions keep their original probe dates and references.

Use the validated local environment and `setup_2/baselines/TiMem/run_local.sh`.
The answer/summary model is `gemini-3.5-flash`; the judge is
`gemini-3.1-pro-preview`. Use one answer worker and one judge request per user.
The schedule contains 2,738 sessions, 120 user/month checkpoints, and 11,721
cumulative answer instances. The adapter profile is `hierarchy_dense_top_k`.

Launch a detached process, verify all five users have started ingesting, and
record its PID, start time, log, results directory, and checkpoint status.
Keep all runtime artifacts inside this repository. Do not change benchmark
code, datasets, other baselines, or agent implementations.

## Comments

- The user explicitly requested this full benchmark after local setup and
  first-month validation. This request authorizes the full Gemini run.
- Detached PID/session `42623` started at `2026-10-05T00:36:25.782729+00:00`.
  All five users were verified actively ingesting and saving memories.
  The launch task is resolved; benchmark completion remains pending.
- Execution metadata and startup evidence are saved in
  `setup_2/tmp/timem_full_run.json`; live output is in
  `setup_2/tmp/timem_local.log` and results are in
  `setup_2/baselines/TiMem/results/v2_timem_local/`.
