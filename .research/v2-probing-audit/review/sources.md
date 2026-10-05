# Independent review sources

Primary evidence remains in the repository. No external sources were used.

- `datasets/v2/u1..u5/probing_questions.json`, `qa_pool.json`, `world_state.json`, `conversations.json`, and selected `sessions/*.json`: questions, immutable payload provenance, daily hidden truth, and observed utterances.
- `generator_v2/DECISIONS.md`: accepted bounded change dates, intended inference, unchanged anchors on reversion, and truth/knowledge distinctions.
- `generator_v2/qa.py`, `probing.py`, `rules.py`, `conversation.py`: gold construction, viewpoint scheduling, rule semantics and writer obligations.
- `experiments/benchmarks/v2.py`, `experiments/core/runner.py`, `grading.py`: actual loader, memory ingestion and metric/grading semantics.
- Root audit `plan.md`, `sources.md`, `manifest.json`, `structure_summary.json`, `calendar_issues.json`, `selection.json`, `summary.json`, `findings.json`, `question_coverage.jsonl`, and `report.md`: scope, evidence inventory, aggregate findings and reconciliation targets.
- Recall `conclusions.md`, `summary.json`, `issues.json`, `coverage.jsonl`, `phase_transcript_review.json`, `boundary_transcript_reviews.json`, `manual_paraphrase_reviews.json`: exhaustive family accounting and focused observation evidence.
- Gold `findings.md`, `summary.json`, `issues.json`, `uncertainties.json`, `coverage.json`, `observation_limits.json`, `manual_evidence.md`: semantic reconstruction, observation limits and focused raw evidence.
- Source `report.md`, `inventory.json`, `flag_reviews.json`, `integration.json`, `native_world_checks.json`, and `verification.json`: source consistency, independently annotated saved flags and offline integration evidence.

The independent review script checks canonical source hashes against the root manifest. Its JSON output records source-derived values and counterfactual outcomes separately from the conclusions in `review.md`.
