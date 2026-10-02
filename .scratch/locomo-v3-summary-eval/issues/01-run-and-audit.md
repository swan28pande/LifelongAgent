# Run and audit the one-conversation live evaluation

Type: task
Status: resolved

Execute [the evaluation specification](../spec.md) with the user's existing Gemini
authorization. Reuse the prepared environment and cached embedding model. Persist
answers and grades independently, verify source hashes and all planning/budget
traces, and report concrete results with timings.

Artifacts: `tmp/locomo-v3-update-summary-eval/`.

## Answer

Completed the fresh run on conv-26 with hierarchical summaries and optional
distillation. Gemini Pro judged 140/199 answers correct (70.4%), compared with
138/199 (69.3%) in the previous run. Answerable: 111/152 (73.0%); adversarial:
29/47 (61.7%).

Runtime: 2747.387 seconds (45m 47s). Retrievals: 547 total, average 2.75, maximum
five; previous total 647 and average 3.25. All 199 dispatch/schema audits passed,
with zero budget violations and no tool/model/ingestion/summary failures. Five
invalid proposals were blocked and corrected. Agent source hashes remained unchanged.

Results, individual grades, runtime breakdown, source/config manifests, independent
verification, and comparison are saved in the artifact directory. The run includes
extraction, summarization, retrieval, and distillation changes and does not isolate
their individual effects.

The combined summary observation was truncated at 12,000 characters in 196/197
lookups, frequently cutting off later weekly details. Overall word-overlap F1
decreased from 0.535 to 0.501. These limitations are recorded in report.md and
analysis.json; application code was not changed during evaluation.
