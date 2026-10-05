# Launch and verify the full benchmark

Type: task
Status: resolved

Launch the existing local TiMem command in a detached session, confirm five
worker processes and ongoing ingestion for every user, and save execution
metadata without changing the evaluation implementation.

## Comments

- Preflight: no existing setup_2 launcher or full-run directory was found.
  The dry-run schedule contains five users and 11,721 cumulative answers;
  existing Google application-default credentials refreshed successfully.

## Answer

Launched at `2026-10-05T00:36:25.782729+00:00` in detached session/PID
`42623`. Verified at `2026-10-05T00:37:05.532297+00:00` that the parent is
alive and five worker processes (`42629`–`42633`) are actively ingesting March
2026. All five users have saved real L1 memories and ingestion ledger entries:
u1 has four sessions; u2, u3, u4, and u5 each have three at verification.

The manifest confirms the original-probe-date cumulative protocol,
`hierarchy_dense_top_k`, `gemini-3.5-flash`, and `gemini-3.1-pro-preview`.
The benchmark is running; no checkpoint accuracy or final result is available
yet. No benchmark code or dataset was changed.

Execution metadata: `setup_2/tmp/timem_full_run.json`.
Log: `setup_2/tmp/timem_local.log`.
Results: `setup_2/baselines/TiMem/results/v2_timem_local/`.

To resume after an interruption, rerun
`bash setup_2/baselines/TiMem/run_local.sh` from the repository root.
Completed checkpoints are skipped; an interrupted memory build replays its
month, and interrupted evaluation retries missing answers or grades.
