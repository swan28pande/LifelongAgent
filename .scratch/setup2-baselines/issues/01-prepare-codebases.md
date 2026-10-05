# Prepare the three baseline codebases

Type: task
Status: resolved

Inspect the local baseline sources, copy their usable implementations into
`setup_2/baselines/`, and record source provenance and runtime requirements.
Keep the monthly runner and method behavior unchanged. Verify the copies offline.

## Comments

- DirectPrompting was explicitly clarified by the user as full context.
- Existing benchmark implementations are in `experiments/methods/`.
- TiMem's local source already includes a Gemini/Vertex AI adapter. Its benchmark
  adapter keeps hierarchy inputs in memory, re-adds summaries during finalization,
  uses collections based on instance ID rather than store directory, and uses a
  fixed day 28 as month end. Preserve and document these inherited limits here.

## Answer

Created separate `TiMem`, `NaiveRAG`, and `DirectPrompting` folders. TiMem includes
262 unchanged source files and the relocated benchmark adapter. NaiveRAG includes
the exact original implementation and its relocated benchmark adapter.
DirectPrompting retains the existing `FullContext` implementation. Shared helper
code was copied; experiment types, prompts, configuration, and embedding behavior
remain shared. Dependency files and preparation/integration limits are documented.

Verified all three adapter imports, unchanged adapter class bodies, compilation of
197 Python files, and source-copy hashes. Offline NaiveRAG/full-context checks
covered sequential ingestion, date-preserving question prompts, and exclusion of
reference answers. All 3,087 protected existing files retained their hashes.
No benchmarks, paid calls, service launches, or monthly-runner edits were made.
