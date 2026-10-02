# Hierarchical summaries with planned retrieval

Status: implemented and verified; 109 offline tests pass.

Bring the summarization update from `memory_v3/` (commit `9eb992a`) into
`memory_v3_update/`. Application changes stay inside `memory_v3_update/`.

- Preserve the existing plan → one guarded retrieval → reassess graph, exact
  committed tool/argument checks, finite repairs, and five-invocation ceiling.
- Integrate per-speaker weekly, monthly, yearly, and lifetime summaries, readable
  narratives, deduplication, updated extraction/pattern prompts, and optional
  themed knowledge distillation.
- Add `semantic_retrieve_memory` to the guarded registry. A combined lookup uses
  one retrieval slot. Prefer it when stored memory is needed, while allowing
  answers from current context without tools and specific follow-up retrievals.
- Reuse stored vectors for layer selection. Retain targeted `get_summary` and
  existing validation, observation limits, and incomplete-evidence guidance.
- Keep summary generation outside the retrieval budget and reader tool registry.
  Preserve `ingest(update_summaries=False)` and `flush(update_summaries=True)`.
- Test generation, persistence, incremental updates, optional distillation, and
  integration with the real compiled reader without external model calls.
- Leave `memory_v3/` unchanged; put all temporary artifacts under project `tmp/`.
