# Correct question grounding and selection

Type: task
Status: resolved

Fix temporal facts, citations, observable date uncertainty, inferred recalls,
completed durations, unknown-answer sampling, current patterns, and partial months.
Regenerate QA and probes without changing ground truth or evaluation semantics.

## Comments

The user confirmed that monthly questions must remain historical.

## Answer

Implemented shared observation grounding and probe-specific reference generation.
All 1,000 regenerated questions pass world-truth/temporal/citation checks. All 29
earlier question findings have recorded resolutions, including 20 boundary-date
witnesses. See `.research/v2-dataset-fixes/finding_resolutions.json` and `tests.json`.
