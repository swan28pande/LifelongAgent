# vm2 results (instance-20261005-171607-swan)

Exported 2026-10-09 before deleting the VM. Memory stores (Mem0 FAISS indexes, Zep/Graphiti Kuzu
graphs, checkpoint backups) are omitted here; the full copy including stores (~1 GB) is kept outside
the repo at `LifelongAgent_vm2_backup/` (plus the archive `vm2_all_2026-10-09.tgz`).

| Folder | Contents |
|---|---|
| `setup_2_mem0/v2_mem0` | Mem0 (mem0ai 2.2.0) continual run, all 5 users × 24 months, gemini-3.5-flash. Final 28.5%, cumulative 33.19% |
| `setup_2_mem0/v2_mem0_pilot` | Mem0 pilot (u5, March 2026) |
| `setup_2_zep/v2_zep_full` | Zep (self-hosted Graphiti on Kuzu) continual run, all 5 users × 24 months, gemini-3.5-flash. Final 32.1%, cumulative 38.11% |
| `setup_2_zep/v2_zep` | Zep pilot (u1, March–April 2026) |
| `full_context_end/u1` | Full context, u1, all 200 probe questions after the full history, one call per question (98.0%) |
| `full_context_batch/u1..u5` | Full context, all 200 probe questions per user in one call per user (76.8% overall) |
| `full_context_batch_v0` | First batch attempt with the output-limit bug; do not use |
| `system_3_smoke` | Early system_3 smoke tests (superseded by the vm3 runs) |
| `logs` | Run logs |

Each setup_2 run folder has `summary.json` (with `by_month`, `by_user`), `manifest.json`, and per user
`months/<YYYY-MM>/{answers,grades,ingest}.jsonl`, `summary.json`, `usage.json`.
